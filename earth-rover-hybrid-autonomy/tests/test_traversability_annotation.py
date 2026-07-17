import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.traversability_annotation import (
    AnnotationCandidate,
    class_id_mask,
    convert_pseudo_seed,
    import_cvat_masks,
    select_annotation_candidates,
    validate_annotation_dataset,
)


def candidate(ride: str, timestamp: float, category: str, frame: int = 0) -> AnnotationCandidate:
    return AnnotationCandidate(
        source_sample_id=f"source_{ride}_{frame}",
        image_path=Path("unused.jpg"),
        pseudo_mask_path=Path("unused.png"),
        ride_id=ride,
        timestamp=timestamp,
        frame_id=frame,
        manifest_index=frame,
        playlist=f"ride_{ride}/front.m3u8",
        segment=f"ride_{ride}/front.ts",
        action_label="FORWARD",
        linear=0.2,
        angular=0.0,
        scene_categories=(category,),
        category_evidence={},
    )


def test_selection_is_deterministic_diverse_and_time_separated() -> None:
    candidates = [
        candidate(str(ride), ride * 100.0 + offset, category, frame=offset)
        for ride, category in enumerate(("PAVED_GROUND", "OFF_ROAD_GROUND", "PERSON"))
        for offset in (0, 2, 8, 16)
    ]

    first = select_annotation_candidates(candidates, 6, 5.0, seed=17)
    repeated = select_annotation_candidates(candidates, 6, 5.0, seed=17)

    assert first == repeated
    assert len(first) == 6
    assert {item.ride_id for item in first} == {"0", "1", "2"}
    assert {item.scene_categories[0] for item in first} == {
        "PAVED_GROUND", "OFF_ROAD_GROUND", "PERSON"
    }
    for ride_id in {item.ride_id for item in first}:
        timestamps = sorted(item.timestamp for item in first if item.ride_id == ride_id)
        assert all(right - left >= 5.0 for left, right in zip(timestamps, timestamps[1:]))


def test_pseudo_seed_mapping_never_creates_off_road(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "destination.png"
    cv2.imwrite(str(source), np.array([[0, 1, 2]], dtype=np.uint8))

    convert_pseudo_seed(source, destination)

    assert cv2.imread(str(destination), cv2.IMREAD_UNCHANGED).tolist() == [[3, 1, 0]]


def test_rgb_contract_mask_converts_to_exact_ids() -> None:
    rgb = np.array([[[0, 0, 0], [38, 166, 91], [43, 126, 216], [220, 50, 47]]], dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result = class_id_mask(bgr, contract_colors())

    assert result.tolist() == [[0, 1, 2, 3]]


def test_validator_accepts_complete_single_channel_dataset(tmp_path: Path) -> None:
    write_annotation_bundle(tmp_path)
    cv2.imwrite(str(tmp_path / "masks/trav_v1_00000.png"), np.array([[0, 1], [2, 3]], dtype=np.uint8))

    report = validate_annotation_dataset(tmp_path)

    assert report["valid"] is True
    assert report["validated_mask_count"] == 1
    assert report["class_pixel_counts"] == {"IGNORE": 1, "ON_ROAD": 1, "OFF_ROAD": 1, "OBSTACLE": 1}


def test_validator_rejects_duplicate_metadata_and_missing_mask(tmp_path: Path) -> None:
    row = write_annotation_bundle(tmp_path)
    with (tmp_path / "metadata.csv").open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=row).writerow(row)

    report = validate_annotation_dataset(tmp_path)

    assert report["valid"] is False
    assert any("mask file is missing" in error for error in report["errors"])
    assert any("duplicate sample_id" in error for error in report["errors"])


def test_validator_rejects_invalid_id_and_dimension_mismatch(tmp_path: Path) -> None:
    write_annotation_bundle(tmp_path)
    cv2.imwrite(str(tmp_path / "masks/trav_v1_00000.png"), np.full((3, 2), 8, dtype=np.uint8))

    report = validate_annotation_dataset(tmp_path)

    assert report["valid"] is False
    assert "dimensions differ" in report["errors"][0]


def test_validator_rejects_invalid_class_id(tmp_path: Path) -> None:
    write_annotation_bundle(tmp_path)
    cv2.imwrite(str(tmp_path / "masks/trav_v1_00000.png"), np.full((2, 2), 8, dtype=np.uint8))

    report = validate_annotation_dataset(tmp_path)

    assert report["valid"] is False
    assert "invalid class IDs" in report["errors"][0]


def test_validator_rejects_multichannel_final_mask(tmp_path: Path) -> None:
    write_annotation_bundle(tmp_path)
    cv2.imwrite(str(tmp_path / "masks/trav_v1_00000.png"), np.zeros((2, 2, 3), dtype=np.uint8))

    report = validate_annotation_dataset(tmp_path)

    assert report["valid"] is False
    assert "single-channel" in report["errors"][0]


def test_validator_rejects_filename_mismatch_and_extra_mask(tmp_path: Path) -> None:
    row = write_annotation_bundle(tmp_path)
    row["mask_path"] = "masks/wrong.png"
    write_metadata_csv(tmp_path / "metadata.csv", row)
    cv2.imwrite(str(tmp_path / "masks/extra.png"), np.zeros((2, 2), dtype=np.uint8))

    report = validate_annotation_dataset(tmp_path)

    assert report["valid"] is False
    assert any("does not match sample_id" in error for error in report["errors"])
    assert any("unexpected mask" in error for error in report["errors"])


def test_validator_rejects_duplicate_source_metadata(tmp_path: Path) -> None:
    first = write_annotation_bundle(tmp_path)
    second = dict(first)
    second.update(
        {
            "sample_id": "trav_v1_00001",
            "image_path": "images/trav_v1_00001.jpg",
            "mask_path": "masks/trav_v1_00001.png",
        }
    )
    cv2.imwrite(str(tmp_path / second["image_path"]), np.zeros((2, 2, 3), dtype=np.uint8))
    (tmp_path / "metadata/trav_v1_00001.json").write_text(
        json.dumps({"sample_id": "trav_v1_00001"}), encoding="utf-8"
    )
    with (tmp_path / "metadata.csv").open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=second).writerow(second)

    report = validate_annotation_dataset(tmp_path, require_masks=False)

    assert report["valid"] is False
    assert any("duplicate source frame metadata" in error for error in report["errors"])


def test_import_cvat_color_mask_and_validate(tmp_path: Path) -> None:
    write_annotation_bundle(tmp_path)
    rgb = np.array([[[0, 0, 0], [38, 166, 91]], [[43, 126, 216], [220, 50, 47]]], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    export = tmp_path / "export.zip"
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("SegmentationClass/trav_v1_00000.png", encoded.tobytes())

    result = import_cvat_masks(tmp_path, export)

    assert result["imported_mask_count"] == 1
    assert validate_annotation_dataset(tmp_path)["valid"] is True
    assert cv2.imread(str(tmp_path / "masks/trav_v1_00000.png"), cv2.IMREAD_UNCHANGED).tolist() == [[0, 1], [2, 3]]


def write_annotation_bundle(root: Path) -> dict[str, str]:
    for directory in ("images", "masks", "metadata"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    contract = {
        "ignore_index": 0,
        "classes": [
            {"id": class_id, "name": name, "color_rgb": list(contract_colors()[class_id])}
            for class_id, name in enumerate(("IGNORE", "ON_ROAD", "OFF_ROAD", "OBSTACLE"))
        ],
    }
    (root / "label_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    cv2.imwrite(str(root / "images/trav_v1_00000.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
    row = {
        "sample_id": "trav_v1_00000",
        "image_path": "images/trav_v1_00000.jpg",
        "mask_path": "masks/trav_v1_00000.png",
        "ride_id": "7",
        "timestamp": "1000.0",
        "frame_id": "4",
        "manifest_index": "12",
        "playlist": "ride_7/front.m3u8",
        "segment": "ride_7/front.ts",
        "source_pseudo_sample_id": "sample_00000",
        "action_label_reference_only": "FORWARD",
        "linear": "0.2",
        "angular": "0.0",
        "scene_categories": "PAVED_GROUND",
        "scene_category_source": "test",
        "review_status": "NOT_ANNOTATED",
    }
    write_metadata_csv(root / "metadata.csv", row)
    (root / "metadata/trav_v1_00000.json").write_text(
        json.dumps({"sample_id": "trav_v1_00000"}), encoding="utf-8"
    )
    return row


def write_metadata_csv(path: Path, row: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)


def contract_colors() -> dict[int, tuple[int, int, int]]:
    return {0: (0, 0, 0), 1: (38, 166, 91), 2: (43, 126, 216), 3: (220, 50, 47)}
