import numpy as np

from flymaze.env import FlyMazeEnv
from flymaze.ppo import SymmetricPPOActionWrapper


def test_symmetric_ppo_wrapper_maps_forward_to_public_range() -> None:
    env = SymmetricPPOActionWrapper(FlyMazeEnv("train_l_left"))
    try:
        np.testing.assert_allclose(env.action(np.asarray([-1.0, -1.0], dtype=np.float32)), [0, -1])
        np.testing.assert_allclose(env.action(np.asarray([0.0, 0.0], dtype=np.float32)), [0.5, 0])
        np.testing.assert_allclose(env.action(np.asarray([1.0, 1.0], dtype=np.float32)), [1, 1])
    finally:
        env.close()
