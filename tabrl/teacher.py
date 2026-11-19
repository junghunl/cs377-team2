"""Teacher: TabPFN regressor(s) over features [s, onehot(a)] whose predictions are Q_T(s, a), and the labels it
is fitted on (label = fqi | dqn).

TabPFN versions: '3.5' = the default model of pip tabpfn 9.x (needs TABPFN_TOKEN in .env);
'2.0.9' = pip tabpfn==2.0.9 (the version ICR-RL uses), which must be installed in its own environment.
"""
import importlib.metadata
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE = {"3.5": "9.", "2.0.9": "2.0.9"}  # teacher.version -> required pip tabpfn version (prefix)


def load_dotenv():
    """TABPFN_TOKEN from the project's .env (never printed)."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def tabpfn(version, n_estimators, device, seed):
    if version not in PACKAGE:
        raise ValueError(f"teacher.version must be one of {list(PACKAGE)}, got '{version}'")
    installed = importlib.metadata.version("tabpfn")
    if not installed.startswith(PACKAGE[version]):
        raise RuntimeError(f"teacher.version {version} needs pip tabpfn {PACKAGE[version]}*, this Python has {installed}")
    load_dotenv()
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(device=device, n_estimators=n_estimators, random_state=seed, fit_mode="fit_with_cache")


class Teacher:
    """Q_T = mean of the predictions of `models` (one model, or two for double FQI)."""

    def __init__(self, n_actions, version="3.5", n_estimators=1, device="cuda", seed=0, chunk=2000):
        self.nA, self.chunk = n_actions, chunk
        self.spawn = lambda: tabpfn(version, n_estimators, device, seed)
        self.models = [self.spawn()]
        self.ready, self.n_fits = False, 0

    def feat(self, s, a):
        return np.concatenate([np.asarray(s, np.float32), np.eye(self.nA, dtype=np.float32)[a]], 1)

    def fit(self, X, y, model=None):
        (model or self.models[0]).fit(X, np.asarray(y, np.float32))
        self.ready = True
        self.n_fits += 1

    def predict(self, X, model=None):
        if model is not None or len(self.models) == 1:
            return np.asarray((model or self.models[0]).predict(X), np.float32)
        return sum(np.asarray(m.predict(X), np.float32) for m in self.models) / len(self.models)

    def q(self, s, model=None):
        """Q_T(s, .) for every state: [N, n_actions] (`model`: one member only)."""
        s = np.asarray(s, np.float32).reshape(-1, np.shape(s)[-1])
        n = len(s)
        X = self.feat(np.repeat(s, self.nA, 0), np.tile(np.arange(self.nA), n))
        rows = self.chunk * self.nA
        return np.concatenate([self.predict(X[i:i + rows], model) for i in range(0, len(X), rows)]).reshape(n, self.nA)


class OutputCache:
    """Q_T(s, .) per buffer row for the loss paths. The teacher only changes at a refit, so the whole buffer is
    evaluated once per refit and new rows in batches; rows not filled yet are masked out (TD loss only)."""

    def __init__(self, capacity, n_actions):
        self.qt = np.zeros((capacity, n_actions), np.float32)
        self.valid = np.zeros(capacity, bool)

    def invalidate(self, i=None):
        if i is None:
            self.valid[:] = False
        else:
            self.valid[i] = False

    def fill(self, buf, teacher, chunk=2000):
        idx = np.flatnonzero(~self.valid[:buf.size])
        for j in range(0, len(idx), chunk):
            k = idx[j:j + chunk]
            self.qt[k] = teacher.q(buf.take(k).s)
            self.valid[k] = True

    def get(self, idx):
        return SimpleNamespace(q=self.qt[idx], mask=self.valid[idx].astype(np.float32))


def clip_bounds(r, gamma):
    """Range of any value when every reward lies in [min(r, 0), max(r, 0)]: [r_lo, r_hi] / (1 - gamma)."""
    return min(float(r.min()), 0.0) / (1 - gamma), max(float(r.max()), 0.0) / (1 - gamma)


class FQI:
    """In-context fitted Q-iteration on the teacher's own Bellman targets (ICR-RL), k backups per refit.

        r~ = r + U(0, noise) once per refit (TabPFN cannot fit constant labels)
        double=True : the context is split at random into halves A and B, one TabPFN each; each half's target
                      uses its own argmax and the other model's value, y_A = r~ + gamma (1-d) Q_B(s', argmax Q_A(s', .))
                      (no max / overestimation bias); Q_T = mean of A and B. False = one TabPFN on the whole
                      context, y = r~ + gamma (1-d) max_a' Q_T(s', a').
        warm_start=True : the previous teacher is Q_0 and the k backups continue from it (the first refit starts
                      from fit(r~)); False = every refit starts from Q_0 = fit(r~) (k + 1 fits).
        clip=True   : every target is clipped to clip_bounds(r) (stops errors compounding over the backups).
    """
    needs_student = False

    def __init__(self, gamma, seed=0, k=10, double=True, warm_start=True, clip=True, noise=1e-4):
        if k < 1:
            raise ValueError("label fqi: k must be >= 1")
        self.gamma, self.k, self.double, self.warm_start, self.clip, self.noise = gamma, k, double, warm_start, clip, noise
        self.rng = np.random.default_rng(seed + 2)

    def fit(self, ctx, teacher, student=None):
        X = teacher.feat(ctx.s, ctx.a)
        r = ctx.r + self.rng.uniform(0, self.noise, len(ctx)).astype(np.float32)
        lo, hi = clip_bounds(ctx.r, self.gamma)
        cont = self.gamma * (1 - ctx.d)
        if not self.double:
            if not (self.warm_start and teacher.ready):
                teacher.fit(X, r)
            for _ in range(self.k):
                y = r + cont * teacher.q(ctx.s2).max(1)
                teacher.fit(X, np.clip(y, lo, hi) if self.clip else y)
            return
        n = len(ctx)
        perm = self.rng.permutation(n)
        hA, hB = perm[: n // 2], perm[n // 2:]
        if len(teacher.models) == 1:
            teacher.models.append(teacher.spawn())
        mA, mB = teacher.models
        if not (self.warm_start and teacher.ready):
            teacher.fit(X[hA], r[hA], model=mA)
            teacher.fit(X[hB], r[hB], model=mB)
        rows = np.arange(n)
        for _ in range(self.k):
            qA2, qB2 = teacher.q(ctx.s2, model=mA), teacher.q(ctx.s2, model=mB)
            yA = r + cont * qB2[rows, qA2.argmax(1)]
            yB = r + cont * qA2[rows, qB2.argmax(1)]
            if self.clip:
                yA, yB = np.clip(yA, lo, hi), np.clip(yB, lo, hi)
            teacher.fit(X[hA], yA[hA], model=mA)
            teacher.fit(X[hB], yB[hB], model=mB)


class DQNLabel:
    """TabQL: the teacher is fitted on the DQN's targets y = r + gamma (1-d) max_a' Q_target(s', a') (one fit)."""
    needs_student = True

    def __init__(self, gamma):
        self.gamma = gamma

    def fit(self, ctx, teacher, student):
        teacher.fit(teacher.feat(ctx.s, ctx.a), ctx.r + self.gamma * (1 - ctx.d) * student.target_value(ctx.s2))


LABELS = {"fqi": FQI, "dqn": DQNLabel}
