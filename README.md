# TabPFN-RL

A DQN agent (the **student**) learns with help from a **teacher** built on TabPFN, a pretrained tabular foundation
model. The teacher does not train any network weights: it estimates Q-values in context, by regressing on a small
set of recent transitions every few hundred steps. The student is an ordinary DQN that acts on its own and also
learns from the teacher's Q-values through an extra loss.

Everything is budgeted in environment steps, so a run with a teacher and a plain DQN run use the same amount of
interaction. Evaluation episodes are not counted.

## How it works

```
                      every 500 steps: refit
 replay buffer ──► context (2,000 transitions) ──► TabPFN teacher Q_T(s, a)
      ▲                                                  │
      │ transitions                                      │ Q_T(s, ·) on the student's minibatch
      │                                                  ▼
 environment ◄────── actions ─────── DQN student ◄── TD loss + margin loss
```

**Teacher.** TabPFN is fitted on rows `[s, onehot(a)] -> target`. Its prediction for every action gives `Q_T(s, ·)`.
The targets come from fitted Q-iteration run inside the context (`label: fqi`): starting from the rewards, the teacher
is refitted `k` times on its own Bellman targets `r + γ max_a' Q_T(s', a')`. The best setting (`conf/best.yaml`)
is **double, rmix, K=10, warm start, margin, clip**:

| part | what it does | switch |
|---|---|---|
| double | split the context into two halves, fit one TabPFN per half, and evaluate each half's targets with the other model (removes the max overestimation); `Q_T` = mean of the two | `label.double` |
| warm start | continue from the previous teacher instead of restarting from the rewards at each refit | `label.warm_start` |
| K = 10 | Bellman backups per refit | `label.k` |
| clip | clip every target to the range any value can take, `[min(r, 0), max(r, 0)] / (1 - γ)` | `label.clip` |
| rmix | context = recent high-return episodes, with half replaced by a uniform sample of the whole buffer at each refit | `context.random_frac` |

**Student.** A DQN (Huber loss, hard target updates, ε-greedy) that chooses every action itself. While the teacher
is in use, its loss gets an extra DQfD large-margin term computed from the teacher's Q-values on each minibatch
state: the action the teacher ranks first should score above the others by a margin. The student thus takes over
the teacher's ranking of actions but learns its own values from TD.

**Gate.** One number `g ∈ [0, 1]` scales every teacher influence. `constant` keeps `g = 1`. `adaptive` compares the
evaluation returns of student and teacher every 5% of the budget; each time the student is strictly better, `g`
halves, and after three passes in a row the teacher is switched off for good.

## Setup

Python 3.11 and a CUDA GPU (a TabPFN run needs about 3-5 GB of GPU memory).

```bash
conda create -n tabpfn-rl python=3.11 -y
conda activate tabpfn-rl
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

TabPFN 3.5 (pip `tabpfn` 9.x) needs an access token. Copy `.env.example` to `.env` and set `TABPFN_TOKEN`.
`.env` is git-ignored.

To use TabPFN 2.0.9 (`teacher.version=2.0.9`), create a second environment named `tabpfn209` next to the first one.
The sweep tool finds it by that name.

```bash
conda create -n tabpfn209 --clone tabpfn-rl -y
conda run -n tabpfn209 pip install tabpfn==2.0.9
```

## Quick start

```bash
# plain DQN on CartPole (ICR-RL setting), 50k steps
python main.py -c conf/envs/icr_cartpole.yaml --run-dir results/demo/dqn seed=0

# DQN + TabPFN teacher (the "best" setting)
python main.py -c conf/envs/icr_cartpole.yaml -c conf/best.yaml --run-dir results/demo/best seed=0

# a shorter run, with the adaptive gate
python main.py -c conf/envs/icr_cartpole.yaml -c conf/best.yaml --run-dir results/demo/adapt \
    train.total_steps=5000 gate.name=adaptive

# TabPFN 2.0.9: run with the Python of the tabpfn209 environment
conda run -n tabpfn209 python main.py -c conf/envs/icr_cartpole.yaml -c conf/best.yaml \
    --run-dir results/demo/tp209 teacher.version=2.0.9
```

A run writes into `--run-dir`:

| file | content |
|---|---|
| `config.yaml` | the full resolved configuration |
| `evals.csv` | one row per evaluation: student return (`eval_mean`), return of the policy that follows the teacher's Q-values (`teacher_eval`, for monitoring only), gate `g`, `max_a Q(s0, a)` and the discounted return actually obtained, for student and teacher |
| `episodes.csv` | one row per training episode |
| `summary.json` | `auc_eval` (mean evaluation return over training, the main metric), `final_eval`, `teacher_off_step`, wall time |
| `model.pt` | the student's Q-network weights |

Evaluation returns are always in the environment's original reward, even when training uses a shaped reward.
Runs are deterministic for a given configuration and seed.

## Configuration

A run is configured by YAML files applied in order (`-c`), then `key=value` overrides.

- Static sections are `env`, `train`, `student` and `teacher` (see `tabrl/config.py`). Unknown keys raise an error.
- Components are `context`, `label`, `refit` and `gate`, each written `{name: ..., params}`, plus `paths`, a list of
  them. A component given in a later file or override replaces the earlier one. On the command line, give `name`
  first, then its parameters: `gate.name=linear gate.decay_steps=0.3`.
- Step parameters of the gates accept a fraction of the budget (`0.05`) or a number of steps (`2500`).

### Options

| section | options |
|---|---|
| `teacher` | `enabled` (false = plain DQN), `version` (`3.5` / `2.0.9`), `n_estimators`, `warmup_episodes`, `min_context` |
| `label` | `{name: fqi, k, double, warm_start, clip}` or `{name: dqn}` (fit the teacher on the DQN's own targets, as in TabQL) |
| `context` | `{name: recent, size, quantile, random_frac}` (episodes whose return reaches the quantile of the context's returns) or `{name: random, size}` (uniform sample of the buffer) |
| `refit` | `{name: transition, every}` (every N steps) or `{name: episode, frac}` (at an episode end once that share of the context is new) |
| `gate` | `constant`; `adaptive` (`kappa`, `check_every`, `check_episodes`, `n_off`, `strict`); `linear` / `exp` (`decay_steps`). All but `constant` take `t_max` (cap `g <= 1 - t / t_max`) and `min_g` |
| `paths` | loss paths: `margin` (`weight`, `margin`; part of best), `mse` (`weight`), `kl` (`weight`, `tau`). Optional acting paths, not part of best: `explore` (`weight`), `dagger` (`weight`, `start`, `end`, `decay`, `kind`, `explore`) |

### Ablations of the best setting

```bash
B="-c conf/envs/icr_cartpole.yaml -c conf/best.yaml"
python main.py $B --run-dir results/abl/no_double  label.double=false
python main.py $B --run-dir results/abl/no_warm    label.warm_start=false
python main.py $B --run-dir results/abl/k1         label.k=1
python main.py $B --run-dir results/abl/no_clip    label.clip=false
python main.py $B --run-dir results/abl/no_rmix    context.random_frac=0
python main.py $B --run-dir results/abl/mse        "paths=[{name: mse, weight: 0.1}]"
python main.py $B --run-dir results/abl/tabql      "label={name: dqn}" "context={name: random}"
```

## Environments

| config | environment | budget | notes |
|---|---|---|---|
| `conf/envs/icr_cartpole.yaml` | CartPole-v1 | 50k | ICR-RL dense training reward `2 - \|x\|/2.4 - \|θ\|/12°`, SB3-style DQN |
| `conf/envs/icr_mountaincar.yaml` | MountainCar-v0 | 150k | ICR-RL reward: progress + 10\|v\| - 1 |
| `conf/envs/icr_acrobot.yaml` | Acrobot-v1 | 50k | ICR-RL reward: tip height |
| `conf/envs/lunarlander.yaml` | LunarLander-v3 | 100k | original reward, DQN hyperparameters of rl-baselines3-zoo |

Any gymnasium environment with a vector observation and discrete actions works: copy one of these files and change
`env.id` and the student settings.

## Running many experiments

`tools/sweep.py` runs environments × variants × seeds as a worker pool and skips runs that already have a
`summary.json`, so an interrupted sweep resumes where it stopped.

```yaml
# save as, for example, conf/sweeps/example.yaml
name: example
seeds: 0-4
envs: [conf/envs/icr_cartpole.yaml]
variants:
  dqn: {files: []}
  best: {files: [conf/best.yaml]}
  best_tp209: {files: [conf/best.yaml], set: {teacher.version: "2.0.9"}, python: tabpfn209}
```

```bash
python tools/sweep.py conf/sweeps/example.yaml --workers 4      # results/example/<env>/<variant>/s<seed>/
python tools/sweep.py conf/sweeps/example.yaml --dry-run        # list the runs
```

`python:` names the conda environment that runs a variant. Keep the number of workers so that their TabPFN runs fit
in GPU memory (four on a 20 GB GPU).

The final experiments are `conf/sweeps/final_a.yaml` and `final_b.yaml`, one per server. `scripts/queue_final.sh`
runs one of them in the background with a retry pass, so it survives logging out:

```bash
PART=a setsid nohup scripts/queue_final.sh > results/queue_final_a.log 2>&1 < /dev/null &
python -I tools/collect.py results OUT_DIR a      # summary files for plotting (runs_a.json, evals_a.json)
```

`scripts/run_tests.sh N FILE` runs a list of short test runs (`name|conda env|arguments` per line) with N in parallel.

## Code layout

```
main.py              run one experiment
tabrl/
  config.py          configuration dataclasses, YAML loading and overrides
  envs.py            environment creation, ICR-RL rewards, greedy evaluation
  data.py            transitions and the replay buffer
  student.py         DQN student
  teacher.py         TabPFN teacher, output cache, labels (fqi, dqn)
  context.py         context selection (recent, random) and refit schedules (transition, episode)
  gate.py            gates (constant, linear, exp, adaptive)
  paths.py           how the teacher reaches the student (loss paths: margin, mse, kl; optional acting paths: explore, dagger)
  trainer.py         the training loop
tools/sweep.py       parallel, resumable sweeps
tools/collect.py     gather finished runs into JSON for plotting
conf/                environments, the best setting, sweeps
scripts/             background queue and test helpers
```

To add a component, write a class in the matching module and add it to that module's registry dictionary
(`LABELS`, `CONTEXTS`, `REFITS`, `GATES` or `PATHS`). Its constructor arguments become its config parameters; the
shared values `gamma`, `seed`, `obs_dim` and `total_steps` are passed to constructors that ask for them.
