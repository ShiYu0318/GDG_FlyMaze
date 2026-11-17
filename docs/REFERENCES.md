# FlyMaze references and traceability

Product behavior is governed by `docs/SPEC.md`; accepted implementation decisions are summarized in `docs/DECISIONS.md`.

## FlyGym 2.1.1

| Primary source | Local use |
|---|---|
| [FlyGym simulation API](https://neuromechfly.org/api_ref/flygym/index.html) | `simulation.py`: reset, warmup, step, state and contact access |
| [Turning-controller tutorial](https://neuromechfly.org/2.1.0/tutorials/4d_turning_controller/) | `simulation.py`, `action.py`: hybrid controller and descending signals |
| [FlyGym anatomy](https://github.com/NeLy-EPFL/flygym/blob/v2.1.1/src/flygym/anatomy.py) | `contact.py`, `wall_collision.py`, `odor.py`: anatomical segment names |
| [FlyGym compose physics](https://github.com/NeLy-EPFL/flygym/blob/v2.1.1/src/flygym/compose/physics.py) | `food.py`, `geometry.py`: explicit contact parameters |
| [FlyGym v1-to-v2 migration](https://github.com/NeLy-EPFL/flygym/blob/v2.1.1/docs/migration.md) | Custom Gymnasium-compatible `env.py` |

## MuJoCo

| Primary source | Local use |
|---|---|
| [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html) | Native model/data access and viewer lifecycle |
| [API functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) | `mj_ray`, `mj_name2id`, `mj_contactForce` |
| [Contact modeling](https://mujoco.readthedocs.io/en/stable/modeling.html#contact) | Actual food/wall contact and explicit pair semantics |
| [Visualization](https://mujoco.readthedocs.io/en/stable/programming/visualization.html) | Passive WSLg viewer in `visualization.py` |

## Gymnasium, Stable-Baselines3, and PyTorch

| Primary source | Local use |
|---|---|
| [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/) | `FlyMazeEnv` spaces and reset/step contract |
| [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) | Training, vector environments, persistence, inference |
| [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | Independent deterministic evaluation and seed reporting |
| [PyTorch installation](https://pytorch.org/get-started/locally/) | CPU wheel index in `pyproject.toml` |

## Reproducibility tooling

| Primary source | Local use |
|---|---|
| [uv project locking](https://docs.astral.sh/uv/concepts/projects/sync/) | `uv.lock`, `uv sync --frozen`, CPU index selection |
| [SciPy sparse solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html) | Static wall-aware odor diffusion in `odor.py` |
| [Microsoft WSL](https://learn.microsoft.com/windows/wsl/install/) | Supported development and WSLg visualization baseline |

## Local review evidence

- `models/ppo_flymaze_mvp.zip` and `models/SHA256SUMS` identify the selected checkpoint.
- `models/ppo_flymaze_mvp.evaluation.json` contains concise training provenance and deterministic success counts.
- `docs/VALIDATION.md` records interpretation and known limitations.
- Unit and integration tests enforce the seven-dimensional observation, success whitelist, held-out split, environment smoke path, and release-model regression.
