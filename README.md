# FlyMaze

以 FlyGym、MuJoCo 與 Stable-Baselines3 PPO 建立的三維果蠅迷宮導航環境。專案固定使用七維感官、二維連續動作，並以頭部或左右實體前足接觸食物作為唯一成功條件。

English documentation follows the Traditional Chinese section.

## 專案狀態

MVP 的物理控制、感官環境與 PPO 管線均已完成。Repository 內附一個經 behavior warm-start、PPO fine-tune 的最終模型：`models/ppo_flymaze_mvp.zip`。

重要限制：最終模型不是純 PPO-from-scratch 的成功案例。純 PPO 訓練流程仍保留供研究與比較；目前正式模型使用三張訓練地圖的 memoryless expert demonstrations 進行 warm-start，再執行 PPO rollout 與 behavior refresh。詳細結果見 [`docs/VALIDATION.md`](docs/VALIDATION.md)。

## 核心契約

Observation 固定為七維 `float32`：

| 索引 | 名稱 | 範圍 | 說明 |
|---:|---|---:|---|
| 0 | `obstacle_front` | `[0, 1]` | 正前方 5 mm 障礙射線；1 表示範圍內無牆 |
| 1 | `obstacle_front_left` | `[0, 1]` | 左前 45° 障礙射線 |
| 2 | `obstacle_front_right` | `[0, 1]` | 右前 45° 障礙射線 |
| 3 | `odor_left` | `[0, 1]` | 左側 `l_funiculus` 的局部氣味濃度 |
| 4 | `odor_right` | `[0, 1]` | 右側 `r_funiculus` 的局部氣味濃度 |
| 5 | `forward_velocity` | `[-1, 1]` | 前向速度除以 15 mm/s 後截斷 |
| 6 | `yaw_rate` | `[-1, 1]` | 偏航角速度除以 3 rad/s 後截斷 |

Action 固定為 `[forward, turn]`：

| 維度 | 範圍 | 說明 |
|---|---:|---|
| `forward` | `[0, 1]` | 前進意圖 |
| `turn` | `[-1, 1]` | 正值向左／正偏航，負值向右 |

Action Adapter 使用：

```text
base = 0.8 * forward
differential = 0.4 * turn
left = clip(base - differential, 0.0, 1.2)
right = clip(base + differential, 0.0, 1.2)
```

Success 白名單永遠只有：

- 頭部 `c_head`
- 左前足 `lf_tarsus5`
- 右前足 `rf_tarsus5`

接觸會在每個 0.2 ms 物理子步檢查，必須具有非正接觸距離與非零接觸力。中後足、胸部、腹部、翅膀或僅靠近食物都不算成功。

## 地圖與資料切分

| Map ID | Split | 路徑 | Timeout |
|---|---|---|---:|
| `train_l_left` | train | 左轉 L 型，28 mm | 10 s |
| `train_l_right` | train | 右轉 L 型，28 mm | 10 s |
| `validation_s_turn_complex` | train auxiliary | 左轉後右轉，40 mm | 14 s |
| `test_straight_unknown` | held-out test | 直線，18 mm | 8 s |

`test_straight_unknown` 不得用於 demonstrations、權重更新或超參數調整。

## 環境建置

支援基準：Ubuntu/WSL2、Python 3.12、uv、FlyGym 2.1.1、CPU-only PyTorch。

```bash
uv sync --frozen
uv run flymaze doctor
uv run pytest -q
```

`uv.lock` 已鎖定 MuJoCo、Gymnasium、Stable-Baselines3、NumPy 與 CPU PyTorch 的相容版本。請勿改用未記錄的 GPU wheel 進行正式比較。

## 自訂地圖：原版使用者的安裝與訓練

**前提：你已依本專案原本的步驟完成 Ubuntu/WSL2、Python 3.12、uv、FlyGym 與 PyTorch 環境安裝，專案位於 `~/GDG_FlyMaze`。** 只需下載擴充 ZIP 與要訓練的 JSON 地圖，沿用既有環境。

| 檔案 | 用途 |
|---|---|
| [FlyMaze_Map_Editor.zip](downloads/FlyMaze_Map_Editor.zip) | 首次安裝 JSON 地圖支援與編輯器 |
| [custom_race_a.json](maps/custom_race_a.json) | 指定地圖 A：8×6 格、最短路線 110 mm |
| [custom_raceb.json](maps/custom_raceb.json) | 指定地圖 B：10×8 格、最短路線 220 mm，含支路 |

開啟 JSON 後按 **Raw** 儲存；ZIP 按 **Download raw file** 下載。ZIP 已包含這兩張地圖，也提供 JSON 方便日後單獨分享。兩張地圖均保留原始的 14 秒回合時限。

### 1. 首次安裝擴充，只需要一次

將 ZIP 下載到 Ubuntu 的 `~/Downloads`：

```bash
cd ~/Downloads
unzip FlyMaze_Map_Editor.zip
python3 FlyMaze_Map_Editor/install_map_editor.py ~/GDG_FlyMaze
```

安裝程式會把必要程式加入原專案、備份原本的地圖載入檔，並放入兩張 JSON。不需要另外下載散裝原始碼或重裝訓練環境。若你已裝過此擴充，可跳過本步。來源版本不同時安裝程式會停止，不會強制覆寫。

### 2. 檢查要訓練的地圖

```bash
cd ~/GDG_FlyMaze
uv run python -m flymaze.map_editor --check maps/custom_race_a.json
uv run python -m flymaze.map_editor --check maps/custom_raceb.json
```

### 3. 選一張指定地圖訓練

地圖 A：

```bash
cd ~/GDG_FlyMaze
FLYMAZE_MAP_FILE="$PWD/maps/custom_race_a.json" \
FLYMAZE_TRAIN_MAP=custom_race_a \
uv run flymaze train-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-custom-race-a
```

地圖 B：

```bash
cd ~/GDG_FlyMaze
FLYMAZE_MAP_FILE="$PWD/maps/custom_raceb.json" \
FLYMAZE_TRAIN_MAP=custom_raceb \
uv run flymaze train-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-custom-race-b
```

`FLYMAZE_MAP_FILE` 指向實際 JSON，`FLYMAZE_TRAIN_MAP` 必須等於其中的 `map_id`。兩個變數只作用於該次命令及其子程序。這是從頭訓練，不是接續既有 checkpoint。

原版程式尚未安裝此擴充時，不會讀取上述兩個變數，仍會使用原本地圖；務必先完成首次安裝。擴充沿用原訓練程式，明確將左彎訓練槽指向你指定的自訂地圖，held-out 地圖維持原樣。

### 4. 之後分享別張地圖，只需 JSON

```bash
cd ~/GDG_FlyMaze
cp ~/Downloads/custom_friend.json maps/

FLYMAZE_MAP_FILE="$PWD/maps/custom_friend.json" \
FLYMAZE_TRAIN_MAP=custom_friend \
uv run flymaze train-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-custom-friend
```

請換成實際檔名與 JSON 裡的 `map_id`。只有要調整訓練參數時才需要改 `configs/ppo.toml`。

要編輯地圖可執行 `uv run python -m flymaze.map_editor --no-browser`，開啟終端機顯示的完整網址。完整操作與還原說明在解壓縮 ZIP 的 `FlyMaze_Map_Editor/docs/MAP_EDITOR.md`。

JSON 格式與路徑連通檢查已通過；完整 FlyGym 模擬與指定地圖的 PPO 訓練仍待 Ubuntu 實測，不保證既有模型能通關。

## 評估內附模型

```bash
uv run flymaze evaluate-ppo \
  --model-path models/ppo_flymaze_mvp.zip \
  --map-id test_straight_unknown \
  --seeds 0 1 2
```

其他合法 map ID 可由上方表格選取。加入 `--output artifacts/evaluation.json` 可輸出 JSON；`artifacts/` 預設不會進入 Git。

## 視覺化

在支援 WSLg 的 Ubuntu shell 執行：

```bash
uv run flymaze visualize-ppo \
  --model-path models/ppo_flymaze_mvp.zip \
  --map-id train_l_left \
  --seed 0 \
  --show-contacts \
  --show-odor \
  --persist
```

若視窗未開啟，先確認 `DISPLAY` 或 `WAYLAND_DISPLAY` 非空，並從 Windows 執行 `wsl --update`、`wsl --shutdown` 後重新開啟 Ubuntu。視覺化只影響顯示，不改變 physics、observation、reward 或成功判定。

## PPO 設定

`configs/ppo.toml` 是正式單一設定來源，兩種訓練命令都會實際讀取它。

共用 PPO 參數：

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `learning_rate` | `0.001` | Adam learning rate |
| `gamma` | `0.99` | 折扣因子 |
| `gae_lambda` | `0.95` | GAE lambda |
| `entropy_coefficient` | `0.005` | entropy bonus |
| `n_steps` | `128` | 每環境 rollout 長度 |
| `batch_size` | `64` | mini-batch；必須整除 `n_envs * n_steps` |
| `network` | `[64, 64]` | MLP hidden layers |
| `device` | `cpu` | 固定 CPU baseline |

純 PPO 預設：4096 requested timesteps、seed 0、4 environments、evaluation seeds 0/1/2。

Multi-task 正式設定：seed 7、demo seed 0、每張圖 128 平衡 samples、2000 warm-start epochs、512 requested PPO timesteps、6 environments、1000 post-PPO behavior epochs。SB3 會完成整個 rollout，因此 512 requested timesteps 實際產生 768 transitions。

### 純 PPO 從零訓練

```bash
uv run flymaze train-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-scratch
```

### 正式 multi-task 流程

```bash
uv run flymaze train-multitask-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-multitask
```

常用 CLI 覆寫包括 `--seed`、`--n-envs`、`--n-steps`、`--batch-size`、`--learning-rate`、`--gamma`、`--gae-lambda`、`--entropy-coefficient` 與 `--network`。純 PPO 另有 `--total-timesteps`；multi-task 另有 `--demonstration-seeds`、`--behavior-epochs`、`--post-ppo-behavior-epochs`、`--ppo-timesteps`、`--samples-per-demonstration`。未提供的值一律取自 TOML。

每次訓練會輸出 checkpoint、TensorBoard logs 與 `training_report.json`。報告包含 resolved configuration、requested/actual transitions、評估結果及模型 SHA-256。

## Environment 與 Reward

- 物理 timestep：0.0002 s
- Agent decision interval：0.1 s（每次 action 執行 500 個物理子步）
- 固定走廊寬度：10 mm
- 氣味網格：0.25 mm
- 氣味參數：`D=100`、`lambda=1`、`S=1`
- Success reward：`+100`
- 每 decision time penalty：`-0.01`
- 發生牆面接觸的 decision penalty：`-0.05`
- 路徑 potential shaping scale：`2.0`，gamma `0.99`

Reward 可用完整路徑距離計算 shaping，但 Agent 永遠只收到七維 observation。正式評估以物理接觸為準，不以 reward 數字判定成功。

## 開發驗證

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python -m compileall -q src
uv build
```

專案架構與規格見 [`docs/SPEC.md`](docs/SPEC.md)，決策摘要見 [`docs/DECISIONS.md`](docs/DECISIONS.md)，外部 API 來源見 [`docs/REFERENCES.md`](docs/REFERENCES.md)。

---

## English

FlyMaze is a three-dimensional embodied navigation environment built with FlyGym, MuJoCo, Gymnasium, and Stable-Baselines3 PPO. It exposes exactly seven observations and two continuous actions. An episode succeeds only when the physical head, left forefoot, or right forefoot contacts the food.

### Project status

The physics, sensory environment, and PPO pipeline are complete. The repository includes `models/ppo_flymaze_mvp.zip`, a behavior-warm-started and PPO-fine-tuned model. It is not evidence of pure PPO-from-scratch mastery. The from-scratch training path remains available for research and comparison.

### Setup

Supported baseline: Ubuntu/WSL2, Python 3.12, uv, FlyGym 2.1.1, and CPU-only PyTorch.

```bash
uv sync --frozen
uv run flymaze doctor
uv run pytest -q
```

### Evaluate the release model

```bash
uv run flymaze evaluate-ppo \
  --model-path models/ppo_flymaze_mvp.zip \
  --map-id test_straight_unknown \
  --seeds 0 1 2
```

### Train PPO from scratch

```bash
uv run flymaze train-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-scratch
```

### Train the multi-task workflow

```bash
uv run flymaze train-multitask-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-multitask
```

`configs/ppo.toml` is loaded by both commands and is the source of truth. CLI flags override only explicitly supplied values. Generated checkpoints, TensorBoard events, reports, and videos belong under `artifacts/` and are ignored by Git.

### Observation and action contracts

The observation is `[front, front-left, front-right, left odor, right odor, forward velocity, yaw rate]`. The first five values are in `[0,1]`; motion values are in `[-1,1]`. No food coordinates, map data, full odor field, or contact status are exposed.

The action is `[forward, turn]`, with `forward` in `[0,1]` and `turn` in `[-1,1]`. Positive turn means left/positive yaw. PPO internally uses a symmetric `[-1,1]` forward output and maps it back to the public range.

### Maps and leakage policy

Training uses `train_l_left`, `train_l_right`, and `validation_s_turn_complex`. `test_straight_unknown` is held out and must never be used for demonstrations, weight updates, or hyperparameter selection.

### Visualization

```bash
uv run flymaze visualize-ppo \
  --model-path models/ppo_flymaze_mvp.zip \
  --map-id train_l_left \
  --show-contacts \
  --show-odor \
  --persist
```

This requires a working MuJoCo display, such as WSLg. Numerical training and evaluation are headless.

### Verification

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python -m compileall -q src
uv build
```

See [`docs/SPEC.md`](docs/SPEC.md) for the authoritative specification and [`docs/VALIDATION.md`](docs/VALIDATION.md) for release-model evidence and known limitations.
