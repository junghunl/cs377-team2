"""Which transitions the teacher is fitted on (context) and when it is refitted (refit)."""
import numpy as np

from .data import Transitions


class Recent:
    """ICR-RL style: an episode joins the context if its return is at least the `quantile` of the returns of
    the episodes currently in the context (warm-up episodes always join); FIFO beyond `size` transitions.
    random_frac = f (> 0): at each refit the oldest f * size rows are replaced by a fresh uniform sample of the
    whole replay buffer (recent good episodes + random coverage; 'rmix'). 0 = off."""

    def __init__(self, obs_dim, size=2000, quantile=0.95, random_frac=0.5, seed=0):
        if not 0 <= random_frac < 1:
            raise ValueError("context recent: random_frac must be in [0, 1)")
        self.size, self.quantile, self.random_frac = size, quantile, random_frac
        self.rng = np.random.default_rng(seed + 1)
        self.data = Transitions.empty(0, obs_dim)
        self.ep = np.zeros(0, np.int64)  # episode id of each context row
        self.hist, self.next_id = {}, 0  # episode id -> return, for the episodes still in the context

    def on_episode_end(self, traj, ret, warmup):
        past = list(self.hist.values())
        if not (warmup or len(past) < 2 or ret >= np.quantile(past, self.quantile)):
            return 0
        eid, self.next_id = self.next_id, self.next_id + 1
        self.data = Transitions.cat(self.data, traj).take(slice(-self.size, None))
        self.ep = np.concatenate([self.ep, np.full(len(traj), eid)])[-self.size:]
        self.hist[eid] = ret
        alive = set(np.unique(self.ep).tolist())
        self.hist = {k: v for k, v in self.hist.items() if k in alive}
        return len(traj)

    def get(self, buf):
        ctx = self.data
        if self.random_frac > 0 and buf.size > 0:
            n_r = min(int(self.size * self.random_frac), buf.size)
            keep = ctx.take(slice(-(self.size - n_r), None)) if len(ctx) > self.size - n_r else ctx
            ctx = Transitions.cat(keep, buf.take(self.rng.choice(buf.size, n_r, replace=False)))
        return ctx

    def reference_size(self):
        return len(self.data)


class Random:
    """TabQL style: a fresh uniform sample of `size` transitions of the replay buffer at every refit."""

    def __init__(self, size=2000, seed=0):
        self.size = size
        self.rng = np.random.default_rng(seed + 1)

    def on_episode_end(self, traj, ret, warmup):
        return len(traj)

    def get(self, buf):
        return buf.take(self.rng.choice(buf.size, min(self.size, buf.size), replace=False))

    def reference_size(self):
        return self.size


CONTEXTS = {"recent": Recent, "random": Random}


class TransitionRefit:
    """Every `every` env steps once the buffer holds learning_starts transitions (TabQL)."""

    def __init__(self, every=500):
        self.every = every

    def at_step(self, st):
        return st.buffer_size >= st.learning_starts and st.steps_since_refit >= self.every

    def at_episode_end(self, st):
        return False


class EpisodeRefit:
    """At the end of an episode once `frac` of the context is new (ICR-RL); the first fit at the end of the
    last warm-up episode."""

    def __init__(self, frac=0.25):
        self.frac = frac

    def at_step(self, st):
        return False

    def at_episode_end(self, st):
        return st.warmup_done and (not st.ready or st.new_since_refit >= self.frac * st.reference_size)


REFITS = {"transition": TransitionRefit, "episode": EpisodeRefit}
