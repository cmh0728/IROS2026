from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml


OFFICIAL_REPOSITORY = "https://github.com/jiaming-ai/GENIE-SAMTP.git"
OFFICIAL_PAPER = "https://arxiv.org/abs/2506.17960"
OFFICIAL_BRANCH = "master"
OFFICIAL_COMMIT = "728aee296cf44288356de683b1948f18b05917d6"
OFFICIAL_MODEL_CONFIG = "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"
OFFICIAL_TRAINING_CONFIG = (
    "sam2/configs/sam2.1_training_tiny/"
    "sam2_training_custom2_freezeNoneNone_f57.yaml"
)
OFFICIAL_CHECKPOINT = (
    "sam2_logs/configs/sam2.1_training_tiny/"
    "sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt"
)
OFFICIAL_CHECKPOINT_URL = (
    "https://drive.google.com/drive/folders/"
    "190yHH-TcfQVoByZeB1809sPIR62CsBD1?dmr=1&ec=wgc-drive-hero-goto"
)
UPSTREAM_LICENSE_STATUS = "no root LICENSE file at the frozen upstream commit"


@dataclass(frozen=True)
class SamTpPrediction:
    """Continuous SAM-TP output for one explicitly RGB uint8 image.

    ``raw_logits`` is the official wrapper's unthresholded mask-logit output.
    ``traversability_score`` is its element-wise sigmoid, where larger values
    mean stronger membership in the bottom-point-prompted traversable region.
    ``heatmap`` is visualization-only RGB and must not be used by a planner.
    """

    raw_logits: np.ndarray
    traversability_score: np.ndarray
    heatmap: np.ndarray
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int]
    inference_time_ms: float
    device: str


def load_reproduction_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SAM-TP reproduction config must be a mapping")
    upstream = config.get("upstream")
    runtime = config.get("runtime")
    if not isinstance(upstream, dict) or not isinstance(runtime, dict):
        raise ValueError("config requires upstream and runtime mappings")
    expected = {
        "repository_url": OFFICIAL_REPOSITORY,
        "branch": OFFICIAL_BRANCH,
        "commit": OFFICIAL_COMMIT,
        "model_config": OFFICIAL_MODEL_CONFIG,
        "training_config": OFFICIAL_TRAINING_CONFIG,
        "checkpoint": OFFICIAL_CHECKPOINT,
    }
    differences = {
        key: {"expected": value, "actual": upstream.get(key)}
        for key, value in expected.items()
        if upstream.get(key) != value
    }
    if differences:
        raise ValueError(f"config differs from the frozen official provenance: {differences}")
    if runtime.get("score_conversion") != "sigmoid":
        raise ValueError("only the explicit sigmoid logit conversion is supported")
    return config


def resolve_upstream_paths(
    upstream_root: str | Path,
    config: dict[str, Any],
    checkpoint_override: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(upstream_root).expanduser().resolve()
    upstream = config["upstream"]
    checkpoint = (
        Path(checkpoint_override).expanduser().resolve()
        if checkpoint_override is not None
        else root / upstream["checkpoint"]
    )
    return {
        "root": root,
        "model_config": root / upstream["model_config"],
        "training_config": root / upstream["training_config"],
        "checkpoint": checkpoint,
    }


def validate_rgb_image(image_rgb: np.ndarray) -> None:
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a NumPy array")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must use uint8 values")
    if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
        raise ValueError("image_rgb dimensions must be non-zero")


def sigmoid_logits(raw_logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(raw_logits, dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError("SAM-TP logits contain NaN or Inf")
    positive = logits >= 0
    result = np.empty_like(logits, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def score_to_heatmap(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("traversability score must be a finite HxW array")
    if values.size and (float(values.min()) < 0.0 or float(values.max()) > 1.0):
        raise ValueError("traversability score must remain in [0, 1]")
    bgr = cv2.applyColorMap(np.rint(values * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def validate_prediction_arrays(
    image_rgb: np.ndarray,
    raw_logits: np.ndarray,
    heatmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    validate_rgb_image(image_rgb)
    logits = np.asarray(raw_logits)
    display = np.asarray(heatmap)
    if not np.issubdtype(logits.dtype, np.floating):
        raise ValueError("SAM-TP logits must use a floating-point dtype")
    if display.dtype != np.uint8:
        raise ValueError("SAM-TP heatmap must use uint8 values")
    if logits.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"SAM-TP output shape {logits.shape} differs from input {image_rgb.shape[:2]}"
        )
    if display.shape != image_rgb.shape:
        raise ValueError(
            f"SAM-TP heatmap shape {display.shape} differs from input {image_rgb.shape}"
        )
    if not np.isfinite(logits).all():
        raise ValueError("SAM-TP logits contain NaN or Inf")
    return logits, display


class SamTpPredictor:
    """Load the official SAM_TP wrapper once and expose validated predictions."""

    def __init__(
        self,
        upstream_root: str | Path,
        model_config: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        loader: Callable[[str, str], Any] | None = None,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        if device != "cuda" and loader is None:
            raise ValueError("the official SAM_TP wrapper is CUDA-only in the frozen commit")
        self.upstream_root = Path(upstream_root).expanduser().resolve()
        self.model_config = Path(model_config).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = device
        self._loader = loader
        self._synchronize = synchronize
        self._model: Any | None = None
        self.load_time_ms: float | None = None
        self.load_count = 0

    def load(self) -> None:
        if self._model is not None:
            return
        for path, label in (
            (self.upstream_root, "upstream repository"),
            (self.model_config, "model config"),
            (self.checkpoint, "checkpoint"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        started = time.perf_counter()
        loader = self._loader or self._official_loader()
        self._model = loader(str(self.model_config), str(self.checkpoint))
        official_model = getattr(self._model, "sam2_model", None)
        if official_model is not None:
            official_model.eval()
        if self._synchronize is not None:
            self._synchronize()
        self.load_time_ms = (time.perf_counter() - started) * 1000.0
        self.load_count += 1

    def _official_loader(self) -> Callable[[str, str], Any]:
        root = str(self.upstream_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            module = importlib.import_module("sam2.sam_tp")
        except ImportError as exc:
            raise RuntimeError(
                "cannot import official sam2.sam_tp; activate the independent "
                "SAM-TP environment and install the upstream checkout with pip install -e"
            ) from exc
        return module.SAM_TP

    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction:
        validate_rgb_image(image_rgb)
        self.load()
        if self._synchronize is not None:
            self._synchronize()
        started = time.perf_counter()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            result = self._model.run_sam2_inference(image_rgb)
        else:
            with torch.inference_mode():
                result = self._model.run_sam2_inference(image_rgb)
        if self._synchronize is not None:
            self._synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(result, dict) or not {"logits", "heatmap"}.issubset(result):
            raise ValueError("official SAM_TP output must contain logits and heatmap")
        logits, official_heatmap = validate_prediction_arrays(
            image_rgb, result["logits"], result["heatmap"]
        )
        score = sigmoid_logits(logits)
        return SamTpPrediction(
            raw_logits=logits,
            traversability_score=score,
            heatmap=official_heatmap,
            input_shape=tuple(int(value) for value in image_rgb.shape),
            output_shape=tuple(int(value) for value in logits.shape),
            inference_time_ms=latency_ms,
            device=self.device,
        )


def compare_state_dicts(
    model_state: dict[str, Any],
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    mismatched = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(value) for value in model_state[key].shape)
        checkpoint_shape = tuple(int(value) for value in checkpoint_state[key].shape)
        if model_shape != checkpoint_shape:
            mismatched.append(
                {
                    "key": key,
                    "model_shape": list(model_shape),
                    "checkpoint_shape": list(checkpoint_shape),
                }
            )
    return {
        "compatible": not missing and not unexpected and not mismatched,
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": mismatched,
    }


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(upstream_root: str | Path) -> dict[str, Any]:
    root = Path(upstream_root).expanduser().resolve()

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
        ).strip()

    return {
        "remote_url": run("remote", "get-url", "origin"),
        "branch": run("branch", "--show-current") or "detached",
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def environment_report(device: str = "cuda") -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "requested_device": device,
    }
    try:
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        report.update(
            {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except ImportError as exc:
        report["torch_import_error"] = str(exc)
    report["ubuntu_version"] = _command_output(["lsb_release", "-ds"])
    query = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    if query:
        parts = [part.strip() for part in query.split(",", 1)]
        report["nvidia_smi_gpu_name"] = parts[0]
        report["nvidia_driver_version"] = parts[1] if len(parts) > 1 else None
    report["nvidia_smi_cuda_compatibility"] = _nvidia_cuda_version()
    packages = {}
    for distribution in (
        "numpy",
        "opencv-python",
        "Pillow",
        "hydra-core",
        "iopath",
        "PyYAML",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    report["packages"] = packages
    return report


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _nvidia_cuda_version() -> str | None:
    output = _command_output(["nvidia-smi"])
    if not output or "CUDA Version:" not in output:
        return None
    return output.split("CUDA Version:", 1)[1].split()[0]


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
