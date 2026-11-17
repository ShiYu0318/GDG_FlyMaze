import numpy as np
import pytest

from flymaze.action import DescendingSignal
from flymaze.simulation import FlySimulation


@pytest.mark.integration
def test_headless_flygym_controller_smoke() -> None:
    with FlySimulation(physics_timestep_seconds=0.0001, seed=0) as simulation:
        initial = simulation.sample_state()
        final = simulation.step_controller(DescendingSignal(1.0, 1.0), substeps=1)

    assert final.simulation_time > initial.simulation_time
    assert np.isfinite(final.thorax_position).all()
    assert np.isfinite(final.thorax_quaternion).all()
