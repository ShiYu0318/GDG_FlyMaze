from flymaze.doctor import environment_report


def test_environment_report_records_required_runtime_versions() -> None:
    report = environment_report()
    packages = report["packages"]
    assert report["python"].startswith("3.12.")
    assert packages["flygym"] == "2.1.1"
    assert packages["mujoco"] != "not-installed"
    assert packages["stable-baselines3"] != "not-installed"
    assert report["torch_cuda_available"] is False
