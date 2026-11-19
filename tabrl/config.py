"""Configuration: YAML files applied in order, then `key=value` overrides.

Static sections (env, train, student, teacher) are dataclasses with strict keys. The swappable parts
(context, label, refit, gate) are `{name: ..., **params}` specs and `paths` is a list of such specs.
A component spec in a later file REPLACES the earlier one (paths: the whole list). On the command line
`gate.name=linear` drops the previous gate's params, so give `name` first and its params after it.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class EnvCfg:
    id: str = "CartPole-v1"
    reward: str = "original"  # original | icr_rl (ICR-RL dense training reward, envs.py)
    target_return: float | None = None  # None -> envs.DEFAULT_TARGETS


@dataclass
class TrainCfg:
    total_steps: int = 50_000  # env-interaction budget (evaluation rollouts are not counted)
    eval_every: int = 500
    eval_episodes: int = 10


@dataclass
class StudentCfg:
    """DQN (target max_a' Q_target(s', a'), Huber loss), SB3-style settings."""
    hidden: list = field(default_factory=lambda: [64, 64])
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 100
    train_freq: int = 1  # env steps between training rounds
    gradient_steps: int = 1  # updates per round (-1 = train_freq)
    target_update: int = 500  # gradient updates between hard target syncs
    grad_clip: float = 10.0
    eps_start: float = 0.95
    eps_end: float = 0.1
    eps_decay_steps: int = 5000


@dataclass
class TeacherCfg:
    enabled: bool = False  # False = plain DQN
    version: str = "3.5"  # TabPFN 3.5 (pip tabpfn 9.x) | 2.0.9 (pip tabpfn==2.0.9, its own conda env)
    n_estimators: int = 1
    warmup_episodes: int = 3  # random episodes at the start (counted in the budget)
    min_context: int = 32  # no refit below this many context transitions
    chunk: int = 2000  # states per TabPFN predict call
    cache_fill_every: int = 100  # env steps between filling new buffer rows into the output cache


COMPONENTS = ("context", "label", "refit", "gate")
SECTIONS = ("env", "train", "student", "teacher")


@dataclass
class Config:
    seed: int = 0
    gamma: float = 0.99
    device: str = "cuda"
    env: EnvCfg = field(default_factory=EnvCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    student: StudentCfg = field(default_factory=StudentCfg)
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    context: Any = field(default_factory=lambda: {"name": "recent"})
    label: Any = field(default_factory=lambda: {"name": "fqi"})
    refit: Any = field(default_factory=lambda: {"name": "transition"})
    gate: Any = field(default_factory=lambda: {"name": "constant"})
    paths: list = field(default_factory=list)

    def to_dict(self):
        return dataclasses.asdict(self)

    def normalize(self):
        for k in COMPONENTS:
            if isinstance(getattr(self, k), str):
                setattr(self, k, {"name": getattr(self, k)})
        self.paths = [{"name": p} if isinstance(p, str) else dict(p) for p in (self.paths or [])]
        return self


def _apply(cfg, d):
    for k, v in (d or {}).items():
        if k in SECTIONS:
            sec = getattr(cfg, k)
            for kk, vv in (v or {}).items():
                if not hasattr(sec, kk):
                    raise KeyError(f"unknown config key '{k}.{kk}'")
                setattr(sec, kk, vv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise KeyError(f"unknown config key '{k}'")


def _set(cfg, dotted, value):
    *parents, key = dotted.split(".")
    obj = cfg
    for p in parents:
        obj = obj[int(p)] if isinstance(obj, list) else obj[p] if isinstance(obj, dict) else getattr(obj, p)
    if isinstance(obj, dict):
        if key == "name" and obj.get("name") != value:
            obj.clear()  # another component: the old params no longer apply
        obj[key] = value
    elif isinstance(obj, list):
        obj[int(key)] = value
    else:
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key '{dotted}'")
        if isinstance(getattr(obj, key), float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        setattr(obj, key, value)


def load_config(files=(), overrides=()):
    cfg = Config()
    for f in files:
        with open(f) as fh:
            _apply(cfg, yaml.safe_load(fh))
    cfg.normalize()
    for o in overrides:
        k, sep, v = o.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got '{o}'")
        _set(cfg, k.strip(), yaml.safe_load(v))
        cfg.normalize()
    return cfg


def build(registry, spec, **shared):
    """spec = {'name': ..., **params}; `shared` values go only to constructors that take them."""
    import inspect
    spec = dict(spec)
    name = spec.pop("name", None)
    if name not in registry:
        raise KeyError(f"unknown component '{name}', available: {sorted(registry)}")
    cls = registry[name]
    accepted = inspect.signature(cls).parameters
    return cls(**spec, **{k: v for k, v in shared.items() if k in accepted})
