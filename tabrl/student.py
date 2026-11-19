"""Student: DQN (target max_a' Q_target(s', a'), Huber loss, hard target syncs).

One update: loss = Huber(Q(s, a), r + gamma (1 - d) max_a' Q_target(s', a')) + sum of the paths' losses
(margin / mse / kl, weighted by weight * g) while the teacher is in use.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(obs_dim, n_actions, hidden):
    layers, d = [], obs_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, n_actions))


class DQN:
    def __init__(self, obs_dim, n_actions, device, gamma, hidden, lr, target_update, grad_clip):
        self.obs_dim, self.dev, self.gamma = obs_dim, device, gamma
        self.target_update, self.grad_clip = target_update, grad_clip
        self.q = mlp(obs_dim, n_actions, hidden).to(device)
        self.q_targ = mlp(obs_dim, n_actions, hidden).to(device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.n_updates = 0

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.dev)

    @torch.no_grad()
    def q_values(self, s):
        return self.q(self._t(s).reshape(-1, self.obs_dim)).cpu().numpy()

    @torch.no_grad()
    def target_value(self, s2):
        """max_a' Q_target(s', a') (the `dqn` label)."""
        return self.q_targ(self._t(s2).reshape(-1, self.obs_dim)).max(1).values.cpu().numpy()

    def update(self, b, paths=(), g=0.0, tv=None):
        """b: Transitions. tv: None or the teacher's outputs for b (q = Q_T(s, .), mask = 1 where they exist)."""
        s, s2, a = self._t(b.s), self._t(b.s2), self._t(b.a, torch.int64)
        r, d = self._t(b.r), self._t(b.d)
        q_all = self.q(s)
        with torch.no_grad():
            y = r + self.gamma * (1 - d) * self.q_targ(s2).max(1).values
        loss = F.smooth_l1_loss(q_all.gather(1, a[:, None])[:, 0], y)
        if tv is not None and g > 0:
            qT, m = self._t(tv.q), self._t(tv.mask)
            for p in paths:
                extra = p.loss(q_all, qT, m)
                if extra is not None:
                    loss = loss + p.weight * g * extra
        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        self.n_updates += 1
        if self.n_updates % self.target_update == 0:
            self.q_targ.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)
