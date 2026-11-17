# FlyMaze release validation

## Release artifact

- Model: `models/ppo_flymaze_mvp.zip`
- SHA-256: `09ea5e841c7d3c0dc237341d4ebb5753cc33ed76310cf53ab18caad7d72f9a1b`
- Policy: SB3 PPO MLP `[64,64]`
- Observation/action: standard seven-dimensional observation and symmetric PPO wrapper over the public two-dimensional action
- Origin: seed-7 multi-task behavior warm-start plus PPO fine-tune

## Training composition

- one successful seed-0 demonstration from each of `train_l_left`, `train_l_right`, and `validation_s_turn_complex`;
- each demonstration balanced to 128 samples, 384 samples total;
- 2,000 behavior warm-start epochs;
- 512 requested PPO timesteps across six environments, 768 actual transitions;
- 1,000 post-PPO behavior-refresh epochs;
- no held-out straight-map sample used for weight updates.

## Deterministic physical evaluation

| Map | Seeds | Success | Successful contacts |
|---|---|---:|---|
| `test_straight_unknown` | 0, 1, 2 | 3/3 | right forefoot, head, right forefoot |
| `train_l_left` | 0, 1, 2 | 2/3 | head, right forefoot |
| `train_l_right` | 0, 1, 2 | 2/3 | right forefoot, right forefoot |
| `validation_s_turn_complex` | 0, 1, 2 | 2/3 | head, head |

All twelve recorded evaluations remained upright. Every successful episode ended via the approved physical contact whitelist. Successful episodes had zero wall-contact Agent steps.

The held-out seed-0 regression is 17 steps, right-forefoot success, zero wall-contact steps. It is exercised by the integration test against the tracked release checkpoint.

## Interpretation

The release demonstrates a functioning PPO pipeline, persistence, deterministic inference, and held-out physical contact. It does not demonstrate robust pure-PPO multi-task learning: preserved development runs from random initialization failed or forgot turning behavior, and the final policy uses supervised behavior warm-start and post-PPO refresh.

## Reproduction commands

```bash
uv sync --frozen
uv run flymaze doctor
uv run flymaze evaluate-ppo \
  --model-path models/ppo_flymaze_mvp.zip \
  --map-id test_straight_unknown \
  --seeds 0 1 2
```

Manual WSLg review:

```bash
uv run flymaze visualize-ppo \
  --model-path models/ppo_flymaze_mvp.zip \
  --map-id train_l_left \
  --seed 0 \
  --show-contacts \
  --show-odor \
  --persist
```
