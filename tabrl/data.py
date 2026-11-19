"""Transitions and the uniform replay buffer shared by the student and the teacher."""
from dataclasses import dataclass, fields

import numpy as np


@dataclass
class Transitions:
    s: np.ndarray  # [N, obs_dim] float32
    a: np.ndarray  # [N] int64
    r: np.ndarray  # [N] float32
    s2: np.ndarray  # [N, obs_dim] float32
    d: np.ndarray  # [N] float32, 1 = terminated (a time-limit truncation is NOT terminal)

    def __len__(self):
        return len(self.a)

    def take(self, idx) -> "Transitions":
        return Transitions(*(getattr(self, f.name)[idx] for f in fields(self)))

    @staticmethod
    def empty(n: int, obs_dim: int) -> "Transitions":
        return Transitions(np.zeros((n, obs_dim), np.float32), np.zeros(n, np.int64),
                           np.zeros(n, np.float32), np.zeros((n, obs_dim), np.float32),
                           np.zeros(n, np.float32))

    @staticmethod
    def from_lists(s, a, r, s2, d) -> "Transitions":
        return Transitions(np.asarray(s, np.float32), np.asarray(a, np.int64), np.asarray(r, np.float32),
                           np.asarray(s2, np.float32), np.asarray(d, np.float32))

    @staticmethod
    def cat(x: "Transitions", y: "Transitions") -> "Transitions":
        return Transitions(*(np.concatenate([getattr(x, f.name), getattr(y, f.name)]) for f in fields(x)))


class Buffer:
    """Uniform ring buffer."""

    def __init__(self, capacity: int, obs_dim: int):
        self.cap = capacity
        self.data = Transitions.empty(capacity, obs_dim)
        self.ptr = self.size = 0

    def add(self, s, a, r, s2, d) -> int:
        i = self.ptr
        x = self.data
        x.s[i], x.a[i], x.r[i], x.s2[i], x.d[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        return i

    def sample_idx(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.size, size=n)

    def take(self, idx) -> Transitions:
        return self.data.take(idx)
