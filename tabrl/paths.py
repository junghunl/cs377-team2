"""Transfer paths: how the teacher reaches the student. Every path has a `weight`; its strength is weight * g.

Acting paths (act(st) -> action or None; st: explore, on, g, t, rng, teacher_action(), student_action()):
    explore : on an epsilon (exploration) step, take the teacher's greedy action w.p. weight * g
    dagger  : w.p. beta(t) = weight * g * schedule(t) the teacher acts, otherwise the student acts greedily
              (explore=True: the student's epsilon-greedy step instead) (Ross et al. 2011)
Loss paths (loss(q_all, qT, m) -> per-batch loss; weighted by weight * g in the student; m = rows with outputs):
    margin  : DQfD large-margin loss on the teacher's greedy action (Hester et al. 2018)
    mse     : mean_a (Q(s, a) - Q_T(s, a))^2
    kl      : KL(softmax(z_T / tau) || softmax(Q)), z_T = Q_T standardized across actions
"""
import torch
import torch.nn.functional as F


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1.0)


class Path:
    def __init__(self, weight=1.0):
        self.weight = weight

    def act(self, st):
        return None

    def loss(self, q_all, qT, m):
        return None


class Explore(Path):
    def act(self, st):
        if st.explore and st.on and st.rng.random() < self.weight * st.g:
            return st.teacher_action()
        return None


def schedule(t, total_steps, start=1.0, end=0.0, decay=0.5, kind="linear"):
    """start -> end; decay = fraction of the budget (<= 1) or steps. linear: reaches `end` at `decay`;
    exp: `decay` is the half-life of (value - end); constant: `start`."""
    d = max(1.0, decay * total_steps if decay <= 1 else decay)
    if kind == "linear":
        return start + (end - start) * min(1.0, t / d)
    if kind == "exp":
        return end + (start - end) * 0.5 ** (t / d)
    if kind == "constant":
        return start
    raise ValueError(f"dagger: kind must be linear, exp or constant, got '{kind}'")


class DAgger(Path):
    def __init__(self, total_steps, weight=1.0, start=1.0, end=0.0, decay=0.5, kind="linear", explore=False):
        super().__init__(weight)
        self.total, self.start, self.end, self.decay, self.kind, self.explore = total_steps, start, end, decay, kind, explore
        schedule(0, total_steps, start, end, decay, kind)  # validate

    def act(self, st):
        if not st.on:
            return None
        if st.rng.random() < self.weight * st.g * schedule(st.t, self.total, self.start, self.end, self.decay, self.kind):
            return st.teacher_action()
        return None if self.explore else st.student_action()


class Margin(Path):
    def __init__(self, weight=1.0, margin=0.8):
        super().__init__(weight)
        self.margin = margin

    def loss(self, q_all, qT, m):
        a_t = qT.argmax(1)
        bonus = self.margin * (1 - F.one_hot(a_t, q_all.shape[1]).to(q_all.dtype))
        return masked_mean((q_all + bonus).max(1).values - q_all.gather(1, a_t[:, None])[:, 0], m)


class MSE(Path):
    def loss(self, q_all, qT, m):
        return masked_mean(((q_all - qT) ** 2).mean(1), m)


class KL(Path):
    def __init__(self, weight=1.0, tau=0.1):
        super().__init__(weight)
        self.tau = tau

    def loss(self, q_all, qT, m):
        p = F.softmax((qT - qT.mean(1, keepdim=True)) / (qT.std(1, keepdim=True) + 1e-6) / self.tau, 1)
        return masked_mean((p * (torch.log(p + 1e-8) - F.log_softmax(q_all, 1))).sum(1), m)


PATHS = {"explore": Explore, "dagger": DAgger, "margin": Margin, "mse": MSE, "kl": KL}
LOSS_PATHS = (Margin, MSE, KL)
