import numpy as np

from flymaze.agents import StatelessRuleAgent


def test_stateless_expert_uses_current_odor_direction_without_memory() -> None:
    agent = StatelessRuleAgent()
    right = agent.predict(np.asarray([0.5, 0.6, 0.4, 0.10, 0.11, 0.0, 0.0], dtype=np.float32))
    left = agent.predict(np.asarray([0.5, 0.4, 0.6, 0.11, 0.10, 0.0, 0.0], dtype=np.float32))
    assert right[1] == -1.0
    assert left[1] == 1.0


def test_stateless_expert_requires_exactly_seven_observations() -> None:
    try:
        StatelessRuleAgent().predict(np.zeros(8, dtype=np.float32))
    except ValueError as exc:
        assert "seven-dimensional" in str(exc)
    else:
        raise AssertionError("eight-dimensional observation was accepted")
