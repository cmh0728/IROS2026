from __future__ import annotations

from typing import Any


CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"
REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
WEIGHT_FILE = "model.safetensors"
ID2LABEL = {0: "ON_ROAD", 1: "OFF_ROAD", 2: "OBSTACLE"}


def build_traversability_segformer(
    pretrained: bool = True,
    config_dict: dict[str, object] | None = None,
) -> Any:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError("install requirements-segmentation.txt on Dell") from exc

    if config_dict is not None:
        config = SegformerConfig.from_dict(config_dict)
    else:
        config = SegformerConfig.from_pretrained(CHECKPOINT, revision=REVISION)
        config.num_labels = 3
        config.id2label = ID2LABEL
        config.label2id = {name: class_id for class_id, name in ID2LABEL.items()}
        config.semantic_loss_ignore_index = 255
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            CHECKPOINT,
            revision=REVISION,
            config=config,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
    return SegformerForSemanticSegmentation(config)
