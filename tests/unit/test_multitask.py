import numpy as np

from flymaze.multitask import MULTITASK_TRAIN_MAP_IDS, public_to_symmetric_action


def test_multitask_training_excludes_held_out_map() -> None:
    assert MULTITASK_TRAIN_MAP_IDS == (
        "train_l_left",
        "train_l_right",
        "validation_s_turn_complex",
    )
    assert "test_straight_unknown" not in MULTITASK_TRAIN_MAP_IDS


def test_public_action_is_mapped_to_symmetric_ppo_space() -> None:
    np.testing.assert_allclose(
        public_to_symmetric_action(np.asarray([0.0, -1.0], dtype=np.float32)), [-1.0, -1.0]
    )
    np.testing.assert_allclose(
        public_to_symmetric_action(np.asarray([1.0, 1.0], dtype=np.float32)), [1.0, 1.0]
    )
