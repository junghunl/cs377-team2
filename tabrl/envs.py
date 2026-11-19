"""Environments (gymnasium, vector observation, Discrete actions).

env.reward = original : gymnasium's reward
env.reward = icr_rl   : the dense training rewards of ICR-RL (Schiff et al., App. B.2), computed from the next
                        observation and replacing the original reward (CartPole, MountainCar, Acrobot).
Evaluation returns are always in original-reward units (info["raw_reward"]), so they stay comparable.
"""
import math

import gymnasium as gym
import numpy as np

DEFAULT_TARGETS = {"CartPole-v1": 475.0, "MountainCar-v0": -110.0, "Acrobot-v1": -100.0, "LunarLander-v3": 200.0}

ICR_RL_REWARDS = {
    # 2 - |x| / x_max - |theta| / theta_max, x_max = 2.4, theta_max = 12 degrees
    "CartPole": lambda o: 2.0 - abs(o[0]) / 2.4 - abs(o[2]) / math.radians(12),
    # (x - x_min) / (x_max - x_min) + 10 |v| - 1, x in [-1.2, 0.6]
    "MountainCar": lambda o: (o[0] + 1.2) / 1.8 + 10.0 * abs(o[1]) - 1.0,
    # tip height -cos(t1) - cos(t1 + t2); obs = [cos t1, sin t1, cos t2, sin t2, ...]
    "Acrobot": lambda o: -o[0] - (o[0] * o[2] - o[1] * o[3]),
}


class Reward(gym.Wrapper):
    def __init__(self, env, fn=None):
        super().__init__(env)
        self.fn = fn

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        info = {**info, "raw_reward": float(r)}
        if self.fn is not None:
            r = self.fn(obs)
        return obs, float(r), term, trunc, info


def make_env(cfg):
    env = gym.make(cfg.id)
    if not isinstance(env.action_space, gym.spaces.Discrete) or not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError(f"{cfg.id}: needs a Box observation and Discrete actions")
    if cfg.reward == "original":
        return Reward(env)
    if cfg.reward == "icr_rl":
        key = next((k for k in ICR_RL_REWARDS if cfg.id.startswith(k)), None)
        if key is None:
            raise ValueError(f"env.reward=icr_rl: no ICR-RL reward for {cfg.id}")
        return Reward(env, ICR_RL_REWARDS[key])
    raise ValueError(f"env.reward must be original or icr_rl, got '{cfg.reward}'")


def target_return(cfg):
    return float(cfg.target_return) if cfg.target_return is not None else DEFAULT_TARGETS.get(cfg.id)


def evaluate(cfg, q_fn, n, seed, gamma):
    """n greedy episodes of q_fn(obs_batch) -> Q[N, A], stepped in lockstep (one q_fn call per time step), on
    separate simulators (not counted in the budget). Returns (returns in original-reward units, discounted
    returns in the training reward, max_a Q(s_0, a) at the start states), each [n]."""
    envs = [make_env(cfg) for _ in range(n)]
    obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    ret, disc, done, k, v0 = np.zeros(n), np.zeros(n), np.zeros(n, bool), 0, None
    while not done.all():
        idx = np.flatnonzero(~done)
        q = q_fn(obs[idx])
        if v0 is None:
            v0 = q.max(1).astype(float)
        for j, a in zip(idx, q.argmax(1)):
            obs[j], r, term, trunc, info = envs[j].step(int(a))
            ret[j] += info["raw_reward"]
            disc[j] += gamma ** k * r
            done[j] = term or trunc
        k += 1
    for e in envs:
        e.close()
    return ret, disc, v0
