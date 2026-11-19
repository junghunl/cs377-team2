"""Training loop: DQN student + optional TabPFN teacher, budgeted in env steps.

Per env step: g = gate(t); act (warm-up episodes random; on an epsilon step the acting paths may act, else a
random action; on a greedy step the acting paths may act, else the student); store; every train_freq steps
`gradient_steps` DQN updates (+ loss paths); refit check (transition schedule); adaptive check; evaluation.
Per episode: the context sees the episode; refit check (episode schedule).
Outputs in the run directory: config.yaml, episodes.csv, evals.csv, summary.json, model.pt.
"""
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .config import build
from .context import CONTEXTS, REFITS
from .data import Buffer, Transitions
from .envs import evaluate, make_env, target_return
from .gate import GATES
from .paths import LOSS_PATHS, PATHS
from .student import DQN
from .teacher import LABELS, OutputCache, Teacher


class Trainer:
    def __init__(self, cfg, run_dir):
        self.cfg = cfg
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            cfg.device = "cpu"
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

        self.env = make_env(cfg.env)
        obs_dim, self.nA = self.env.observation_space.shape[0], int(self.env.action_space.n)
        sc, tc = cfg.student, cfg.teacher
        self.student = DQN(obs_dim, self.nA, cfg.device, cfg.gamma, sc.hidden, sc.lr, sc.target_update, sc.grad_clip)
        self.buf = Buffer(sc.buffer_size, obs_dim)
        shared = dict(gamma=cfg.gamma, seed=cfg.seed, obs_dim=obs_dim, total_steps=cfg.train.total_steps)
        self.gate = build(GATES, cfg.gate, **shared)
        self.paths = [build(PATHS, p, **shared) for p in cfg.paths]
        self.loss_paths = [p for p in self.paths if isinstance(p, LOSS_PATHS)]
        if tc.enabled:
            self.teacher = Teacher(self.nA, tc.version, tc.n_estimators, cfg.device, cfg.seed, tc.chunk)
            self.context = build(CONTEXTS, cfg.context, **shared)
            self.label = build(LABELS, cfg.label, **shared)
            self.refit_rule = build(REFITS, cfg.refit, **shared)
            self.cache = OutputCache(sc.buffer_size, self.nA) if self.loss_paths else None
        else:
            if self.paths:
                raise ValueError("paths need a teacher (teacher.enabled: true)")
            self.teacher = self.cache = None
            self.gate.g = 0.0
        self.n_warm = tc.warmup_episodes if tc.enabled else 0
        self.t = self.episode = 0
        self.last_refit_t = self.new_since_refit = self.n_refits = 0
        self.evals = []

    def teacher_on(self):
        return self.teacher is not None and self.teacher.ready and not self.gate.off and self.gate.g > 0

    def epsilon(self):
        c = self.cfg.student
        return c.eps_start + min(1.0, self.t / max(1, c.eps_decay_steps)) * (c.eps_end - c.eps_start)

    def act(self, obs, explore):
        if self.episode <= self.n_warm:
            return int(self.rng.integers(self.nA))
        st = SimpleNamespace(explore=explore, on=self.teacher_on(), g=self.gate.g, t=self.t, rng=self.rng,
                             teacher_action=lambda: int(self.teacher.q(obs[None])[0].argmax()),
                             student_action=lambda: int(self.student.q_values(obs)[0].argmax()))
        for p in self.paths:
            a = p.act(st)
            if a is not None:
                return a
        if explore:
            return int(self.rng.integers(self.nA))
        return int(self.student.q_values(obs)[0].argmax())

    def learn(self):
        sc = self.cfg.student
        if self.buf.size < max(1, sc.learning_starts) or self.t % sc.train_freq:
            return
        use_t = bool(self.loss_paths) and self.teacher_on()
        for _ in range(sc.train_freq if sc.gradient_steps < 0 else sc.gradient_steps):
            idx = self.buf.sample_idx(sc.batch_size, self.rng)
            self.student.update(self.buf.take(idx), self.loss_paths, self.gate.g, self.cache.get(idx) if use_t else None)

    def refit_state(self):
        return SimpleNamespace(ready=self.teacher.ready, warmup_done=self.episode >= self.n_warm,
                               new_since_refit=self.new_since_refit, reference_size=self.context.reference_size(),
                               steps_since_refit=self.t - self.last_refit_t, buffer_size=self.buf.size,
                               learning_starts=self.cfg.student.learning_starts)

    def can_refit(self):
        return self.teacher is not None and not self.gate.off

    def refit(self):
        ctx = self.context.get(self.buf)
        if len(ctx) < self.cfg.teacher.min_context:
            return
        self.label.fit(ctx, self.teacher, self.student)
        self.last_refit_t, self.new_since_refit = self.t, 0
        self.n_refits += 1
        if self.cache is not None:  # the teacher changed: its cached outputs are stale
            self.cache.invalidate()
            if self.teacher_on():
                self.cache.fill(self.buf, self.teacher, self.cfg.teacher.chunk)

    def run(self):
        cfg, tc = self.cfg, self.cfg.teacher
        ep_f, ev_f = open(self.dir / "episodes.csv", "w", newline=""), open(self.dir / "evals.csv", "w", newline="")
        ep_w, ev_w = csv.writer(ep_f), csv.writer(ev_f)
        ep_w.writerow(["episode", "step", "return", "length", "eps", "g", "refit", "raw_return"])
        ev_w.writerow(["step", "episode", "eval_mean", "eval_std", "teacher_eval", "g",
                       "student_v0", "student_disc", "teacher_v0", "teacher_disc"])
        t0 = time.time()
        while self.t < cfg.train.total_steps:
            self.episode += 1
            obs, _ = self.env.reset(seed=cfg.seed * 100_000 + self.episode)
            traj = {k: [] for k in ("s", "a", "r", "s2", "d")}
            ret, raw, k, done, refits_before = 0.0, 0.0, 0, False, self.n_refits
            while not done and self.t < cfg.train.total_steps:
                self.t += 1
                if self.teacher is not None:
                    self.gate.update(self.t)
                a = self.act(obs, self.rng.random() < self.epsilon())
                obs2, r, term, trunc, info = self.env.step(a)
                done = term or trunc
                i = self.buf.add(obs, a, r, obs2, float(term))
                if self.cache is not None:
                    self.cache.invalidate(i)
                for key, v in zip(traj, (obs, a, r, obs2, float(term))):
                    traj[key].append(v)
                ret, raw, obs, k = ret + r, raw + info["raw_reward"], obs2, k + 1

                if self.cache is not None and self.teacher_on() and self.t % tc.cache_fill_every == 0:
                    self.cache.fill(self.buf, self.teacher, tc.chunk)
                self.learn()
                if self.can_refit() and self.refit_rule.at_step(self.refit_state()):
                    self.refit()
                if self.gate.needs_comparison and self.teacher_on() and self.t % self.gate.check_every == 0:
                    n, g = self.gate.check_episodes, cfg.gamma
                    r_s = evaluate(cfg.env, self.student.q_values, n, 7_000_000 + self.t, g)[0].mean()
                    r_t = evaluate(cfg.env, self.teacher.q, n, 7_000_000 + self.t, g)[0].mean()
                    self.gate.compare(r_s, r_t)
                    self.gate.update(self.t)
                if self.teacher is not None and not self.gate.off and self.gate.g == 0 and not self.gate.never_off:
                    self.gate.off_step = self.t
                if self.t % cfg.train.eval_every == 0:
                    self.log_eval(ev_w, ev_f)
            if self.can_refit():
                x = Transitions.from_lists(*traj.values())
                self.new_since_refit += self.context.on_episode_end(x, ret, self.episode <= self.n_warm)
                if self.refit_rule.at_episode_end(self.refit_state()):
                    self.refit()
            ep_w.writerow([self.episode, self.t, ret, k, round(self.epsilon(), 4), round(self.gate.g, 4),
                           int(self.n_refits > refits_before), raw])
            ep_f.flush()
        if not self.evals or self.evals[-1][0] != self.t:
            self.log_eval(ev_w, ev_f)
        ep_f.close()
        ev_f.close()
        return self.write_summary(time.time() - t0)

    def log_eval(self, w, f):
        """Greedy evaluation of the student (and of the teacher while it is in use). *_v0 = max_a Q(s_0, a) at the
        start states, *_disc = the discounted training-reward return the same policy obtains (scale of Q)."""
        n, g = self.cfg.train.eval_episodes, self.cfg.gamma
        res, s_disc, s_v0 = evaluate(self.cfg.env, self.student.q_values, n, 9_000_000, g)
        t_eval = t_v0 = t_disc = ""
        if self.teacher is not None and self.teacher.ready and not self.gate.off:
            t_ret, t_d, t_q = evaluate(self.cfg.env, self.teacher.q, n, 9_000_000, g)
            t_eval, t_disc, t_v0 = float(t_ret.mean()), float(t_d.mean()), float(t_q.mean())
        self.evals.append((self.t, float(res.mean())))
        w.writerow([self.t, self.episode, float(res.mean()), float(res.std()), t_eval, round(self.gate.g, 4),
                    float(s_v0.mean()), float(s_disc.mean()), t_v0, t_disc])
        f.flush()

    def write_summary(self, wall):
        means = [m for _, m in self.evals]
        target = target_return(self.cfg.env)
        summary = dict(
            env=self.cfg.env.id, seed=self.cfg.seed, steps=self.t, episodes=self.episode,
            final_eval=means[-1], auc_eval=float(np.mean(means)), target=target,
            steps_to_target=next((s for s, m in self.evals if target is not None and m >= target), None),
            teacher_version=self.cfg.teacher.version if self.teacher else None, teacher_off_step=self.gate.off_step,
            teacher_refits=self.n_refits, teacher_fits=self.teacher.n_fits if self.teacher else 0,
            wall_time_s=round(wall, 1))
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.student.save(self.dir / "model.pt")
        return summary
