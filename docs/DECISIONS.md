# FlyMaze accepted decisions

本文件整併先前 ADR-001 至 ADR-037 的現行結論。逐次 sweep 報告與被取代候選已從 production repository 移除；保留的數值均可由測試、release evidence 或既有驗收結果追溯。

## 平台與 API

- 使用 Windows 11 + WSL2 Ubuntu 開發基準、Python 3.12、uv、FlyGym 2.1.1、MuJoCo 與 CPU PyTorch。
- `FlyMazeEnv` 是 Agent 的 Gymnasium 邊界。
- observation 固定七維，action 固定二維。
- 同程序 Python API 僅供可信任研究環境，不是防作弊沙箱。

## 控制與物理

- 正 turn 表示左轉與正 yaw。
- Action Adapter 固定使用 0.8 forward gain、0.4 turn gain、`[0,1.2]` clipping。
- Agent interval 為 100 ms；低階 controller 與 MuJoCo 每個 0.2 ms 子步更新。
- 走廊固定 10 mm；不得按 map、seed 或難度改變。
- 牆壁碰撞包含 rigid body 與六足 locomotor envelope，排除 antenna、wing、haltere。

## 成功、感測與氣味

- 成功只允許 `c_head`、`lf_tarsus5`、`rf_tarsus5`。
- 使用現有 FlyGym mesh 與 explicit food contact pair，不新增擴張白名單的感測 geom。
- 實際接觸在每個物理子步 latch，必須有非正 contact distance 與非零 force。
- 障礙射線為 0°、+45°、-45°，範圍 5 mm，只看 maze walls。
- 氣味網格為 0.25 mm，diffusion length 10 mm，使用 `D=100`、`lambda=1`、`S=1`。

## PPO 與評估

- PPO policy wrapper 將 symmetric forward action 映射到公開 `[0,1]`。
- Collision penalty 由候選 `-0.5` 調整為 `-0.05`，potential shaping scale 為 2.0。
- Learning rate 為 `1e-3`，entropy coefficient 為 `0.005`。
- Held-out straight map timeout 為 8 s。
- 最終模型選擇 seed-7 multi-task policy：三張 training maps 的 memoryless expert demonstrations、behavior warm-start、PPO rollout、post-PPO behavior refresh。
- Held-out map 沒有參與 demonstrations 或權重更新。
- 最終模型不得宣稱為純 PPO-from-scratch success。

## Reopen 規則

變更固定 action/observation、物理 timestep、感測範圍、走廊寬度、氣味參數、成功白名單或 held-out split 前，必須先加入 regression、提出新量測證據並更新本文件與 `SPEC.md`。
