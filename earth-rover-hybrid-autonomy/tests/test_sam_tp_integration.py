from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.sam_tp_reproduction import SamTpPredictor


def test_official_checkpoint_single_image_cuda_smoke() -> None:
    upstream = Path(os.environ.get("SAM_TP_UPSTREAM_ROOT", "")).expanduser()
    checkpoint = Path(os.environ.get("SAM_TP_CHECKPOINT", "")).expanduser()
    image_path = Path(os.environ.get("SAM_TP_SMOKE_IMAGE", "")).expanduser()
    if not all(
        (
            os.environ.get("SAM_TP_UPSTREAM_ROOT"),
            os.environ.get("SAM_TP_CHECKPOINT"),
            os.environ.get("SAM_TP_SMOKE_IMAGE"),
        )
    ):
        pytest.skip("SAM-TP integration paths are not configured")
    model_config = (
        upstream / "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"
    )
    if not all(path.is_file() for path in (model_config, checkpoint, image_path)):
        pytest.skip("SAM-TP integration checkpoint or input is unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image_bgr is not None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor = SamTpPredictor(
        upstream,
        model_config,
        checkpoint,
        synchronize=torch.cuda.synchronize,
    )

    prediction = predictor.predict(image_rgb)

    assert prediction.raw_logits.shape == image_rgb.shape[:2]
    assert prediction.traversability_score.shape == image_rgb.shape[:2]
    assert np.isfinite(prediction.raw_logits).all()
    assert np.isfinite(prediction.traversability_score).all()
    assert predictor.load_count == 1
