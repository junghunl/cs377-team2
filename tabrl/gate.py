"""Gate g in [0, 1]: scales every teacher path (explore / dagger probabilities, loss weights).

    constant : g = 1, never switched off
    linear   : g = 1 - t / decay_steps
    exp      : g = 0.5 ** (t / decay_steps)            (decay_steps = half-life)
    adaptive : every check_every steps the greedy returns of student and teacher are compared (check_episodes each,
               not counted in the budget); a pass (student > teacher; strict=False: >=) multiplies g by kappa,
               n_off passes in a row switch the teacher off; a failed check resets the count.
Common options: t_max > 0 caps g <= 1 - t / t_max (0 = no cap); g < min_g counts as 0. Once g is 0 the teacher is
switched off for good (no more refits). Step parameters accept a float in (0, 1] = fraction of the budget.
"""


def steps(x, total):
    return int(round(x * total)) if isinstance(x, float) and 0 < x <= 1 else int(x)


class Gate:
    needs_comparison = False
    never_off = False

    def __init__(self, total_steps, t_max=0, min_g=0.02):
        self.t_max, self.min_g = steps(t_max, total_steps), min_g
        self.g, self.off_step = 1.0, None

    def schedule(self, t):
        return 1.0

    def update(self, t):
        if self.off_step is not None:
            self.g = 0.0
            return 0.0
        g = self.schedule(t)
        if not self.never_off:
            if self.t_max > 0:
                g = min(g, max(0.0, 1 - t / self.t_max))
            if g < self.min_g:
                g = 0.0
        self.g = g
        return g

    @property
    def off(self):
        return self.off_step is not None


class Constant(Gate):
    never_off = True

    def __init__(self, total_steps):
        super().__init__(total_steps)


class Linear(Gate):
    def __init__(self, total_steps, decay_steps=0.5, **kw):
        super().__init__(total_steps, **kw)
        self.decay = max(1, steps(decay_steps, total_steps))

    def schedule(self, t):
        return max(0.0, 1 - t / self.decay)


class Exp(Gate):
    def __init__(self, total_steps, decay_steps=0.25, **kw):
        super().__init__(total_steps, **kw)
        self.half = max(1, steps(decay_steps, total_steps))

    def schedule(self, t):
        return 0.5 ** (t / self.half)


class Adaptive(Gate):
    needs_comparison = True

    def __init__(self, total_steps, kappa=0.5, check_every=0.05, check_episodes=5, n_off=3, strict=True, **kw):
        super().__init__(total_steps, **kw)
        self.kappa, self.n_off, self.check_episodes, self.strict = kappa, n_off, check_episodes, strict
        self.check_every = max(1, steps(check_every, total_steps))
        self.base, self.passes = 1.0, 0

    def schedule(self, t):
        return self.base

    def compare(self, r_student, r_teacher):
        if r_student > r_teacher or (not self.strict and r_student == r_teacher):
            self.base *= self.kappa
            self.passes += 1
            if self.passes >= self.n_off:
                self.base = 0.0
        else:
            self.passes = 0


GATES = {"constant": Constant, "linear": Linear, "exp": Exp, "adaptive": Adaptive}
