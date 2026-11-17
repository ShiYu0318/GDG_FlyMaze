# FlyMaze MVP v0.1 技術規格

| 欄位 | 內容 |
|---|---|
| 文件版本 | 0.1-final.2 |
| 專案版本 | FlyMaze 0.1.0 |
| 狀態 | 已實作；M0、M1、M2 工程驗收完成 |
| 最後整理日期 | 2026-11-17 |
| 權威性 | 本文件為產品與驗收規格的唯一真實來源 |

## 1. 目標與限制

FlyMaze 是以 FlyGym 2.1.1、MuJoCo、Gymnasium 與 Stable-Baselines3 PPO 建立的三維具身導航環境。MVP 證明物理控制、七維感官、迷宮碰撞、氣味、食物接觸、PPO 訓練、模型儲存載入與 held-out 評估可形成可重現的完整流程。

非目標包括完整果蠅腦模型、高解析視覺、動態氣味流體、正式競賽沙箱、排行榜、GPU 保證，以及 PPO 必須超越規則式策略。

最終 release model 是 behavior-warm-started、PPO-fine-tuned policy，不得描述為純 PPO-from-scratch mastery。

## 2. 鎖定平台

- Python `>=3.12,<3.13`
- uv 與版本化 `uv.lock`
- FlyGym `2.1.1`
- CPU-only PyTorch index
- MuJoCo、Gymnasium、Stable-Baselines3、NumPy 與 SciPy 由 lockfile 鎖定
- Ubuntu/WSL2 是已驗證開發平台；數值流程必須能 headless 執行

## 3. Agent 介面

### 3.1 Observation

標準 observation 必須永遠為七維 `float32`：

```text
[d_front, d_front_left, d_front_right, odor_left, odor_right,
 forward_velocity, yaw_rate]
```

- 三個障礙值與兩個氣味值的範圍為 `[0,1]`。
- 前向速度與偏航角速度的範圍為 `[-1,1]`。
- 障礙值 0 表示射線原點附近，1 表示 5 mm 範圍內無牆。
- 氣味必須在實際左右 `funiculus` 世界座標分別取樣。
- 不得加入接觸、食物座標、目標方向、完整地圖、最短路徑或完整氣味場。

目前速度正規化實作值為 15 mm/s，偏航正規化實作值為 3 rad/s。兩者是已測試的相容基準，但仍屬 `[待校準]`，不得在無新證據與使用者決定下改稱研究門檻。

### 3.2 Action

公開 action 必須為二維：

```text
[forward in 0..1, turn in -1..1]
```

正 turn 表示左轉／正偏航。標準映射固定為：

```text
base = 0.8 * forward
differential = 0.4 * turn
left = clip(base - differential, 0.0, 1.2)
right = clip(base + differential, 0.0, 1.2)
```

PPO 可以在 wrapper 內使用對稱 `[-1,1]` forward 輸出，但進入環境前必須轉回公開 action 契約。

## 4. 物理與成功條件

- 物理 timestep：0.0002 s。
- Agent decision interval：0.1 s；每個 action 執行 500 個低階物理／控制步。
- 走廊內寬固定為 10.0 mm。
- 所有 Agent 共用 FlyGym Hybrid Turning Controller。
- 普通牆面碰撞只記錄並影響 reward，不直接終止 episode。

唯一 success 白名單：

| Role | FlyGym segment |
|---|---|
| head | `c_head` |
| left forefoot | `lf_tarsus5` |
| right forefoot | `rf_tarsus5` |

每個物理子步必須檢查食物 explicit contact pair。只有 `distance <= 0` 且接觸力大於零才成功；事件只 latch 一次並立即 `terminated=true`。任何其他身體部位或無物理接觸的近距離都不得成功。

食物目前為固定地面、半徑 0.5 mm 的 sphere，pair margin 為 0。此幾何已通過 release 行為測試，但研究層級的尺寸／材質敏感度仍為 `[待校準]`。

## 5. 感測與氣味

- 三個障礙射線由實際 head position 發出。
- heading 使用 `c_abdomen12` 指向 `c_head` 的水平向量。
- 方向為 0°、+45°、-45°，最大距離固定 5.0 mm。
- 射線只偵測 maze-wall geom group。
- 氣味場使用 0.25 mm free-space grid。
- 穩態參數固定為 `D=100`、`lambda=1`、`S=1`，正規化至 `[0,1]`。
- 牆格不參與求解；不得用穿牆的直線氣味捷徑。

## 6. 地圖與資料治理

| Map | Split | Route | Timeout |
|---|---|---:|---:|
| `train_l_left` | train | 28 mm | 10 s |
| `train_l_right` | train | 28 mm | 10 s |
| `validation_s_turn_complex` | train auxiliary | 40 mm | 14 s |
| `test_straight_unknown` | held-out test | 18 mm | 8 s |

`test_straight_unknown` 不得用於 demonstrations、權重更新、reward 設計或超參數選擇。正式 L1 難度的通用機器判定規則仍為 `[待校準]`；現有四張圖是版本化 MVP fixtures，不得把它們延伸解讀為完整 L1 taxonomy。

## 7. Episode 與 Reward

正式結果：

- Success：白名單部位實際接觸食物，`terminated=true`。
- Fallen：目前實作為 thorax up-z 小於零連續兩個 Agent steps，`terminated=true`。
- Timeout：達到每圖上限且尚未 terminated，`truncated=true`。
- Simulation error：必須與正常失敗分開處理。

Fallen 的兩步規則是目前實作值，研究層級恢復等待時間仍為 `[待校準]`。

Reward 固定實作：

```text
2.0 * (0.99 * phi_next - phi_previous)
- 0.01 per Agent step
- 0.05 when the Agent step contains wall contact
+ 100 once on success
```

`phi` 由可通行 polyline route 的進展計算。Reward 只供訓練，不得取代實際物理 success。

## 8. PPO 流程

專案必須保留：

1. 純 PPO 隨機初始化訓練、儲存、載入與評估。
2. Memoryless expert demonstrations 的 multi-task behavior warm-start。
3. PPO fine-tune 與 post-PPO behavior refresh。
4. 使用固定 observation/action 的 deterministic checkpoint evaluation。

正式設定由 `configs/ppo.toml` 載入。CLI 只覆寫明確提供的值，報告必須保存 resolved config、requested/actual transitions、模型 hash 與 evaluation seeds。

正式 multi-task 訓練地圖僅為 left、right、S-turn；held-out straight 不得進入訓練。release checkpoint 必須位於 `models/`，生成的 checkpoints 與 logs 必須位於 Git-ignored output directory。

## 9. 驗收與測試

每次改動至少必須通過：

- action、七維 observation、success whitelist 與 map split 單元測試；
- FlyGym headless reset/step integration；
- release checkpoint load 與 held-out deterministic regression；
- Ruff、Mypy、Pytest、compileall 與 wheel/sdist build；
- `uv lock --check` 與 frozen sync。

改變既有行為前必須先新增 regression test。generated experiment artifacts 不得進 Git，除非是使用者明確選定的小型 review evidence。

## 10. 尚未正式收斂的項目

以下項目不得只因目前程式已有常數就宣告為研究門檻：

- 速度與 yaw rate 正規化尺度的分布校準；
- 通用 L1 taxonomy 與難度界線；
- 食物尺寸／材質敏感度；
- Fallen 恢復等待時間與 collision event 去抖定義；
- 多訓練 seeds、每模型 episode 數與「可量測改善」統計門檻；
- 跨硬體或跨執行緒的 bitwise determinism。

這些項目若要正式收斂，必須提出新證據、更新決策紀錄並取得使用者決定。
