from pathlib import Path

from earth_rover.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_latency_2s_profile_loads_without_changing_default_file():
    config = load_config(ROOT / "configs/default.yaml", ROOT / "configs/urban_latency_2s.yaml")

    assert config["project"]["profile"] == "latency_2s"
    assert config["latency"]["sensor_delay_sec"] == 2.0
    assert config["latency"]["frame_delay_sec"] == 2.0
    assert config["latency"]["data_delay_sec"] == 2.0
    assert config["control"]["linear_max"] == 0.22
    assert config["safety"]["frame_timeout_sec"] == 3.0
