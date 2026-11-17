from pathlib import Path

import pytest
from stable_baselines3 import PPO

from flymaze.evaluation import evaluate_agent
from flymaze.ppo import SB3AgentAdapter

PROJECT_ROOT = Path(__file__).parents[2]
MODEL_PATH = PROJECT_ROOT / "models/ppo_flymaze_mvp.zip"


@pytest.mark.integration
def test_selected_ppo_checkpoint_keeps_held_out_seed_zero_behavior() -> None:
    model = PPO.load(MODEL_PATH, device="cpu")
    report = evaluate_agent("test_straight_unknown", SB3AgentAdapter(model), (0,))

    assert report["success_count"] == 1
    result = report["results"][0]
    assert result["success_role"] == "right_forefoot"
    assert result["steps"] == 17
    assert result["wall_contact_steps"] == 0
