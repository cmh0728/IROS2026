from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from training.traversability_review import (
    colorize_mask,
    label_contract,
    write_contact_sheet,
    write_json,
)


ANNOTATION_FIELDS = (
    "sample_id",
    "image_path",
    "mask_path",
    "ride_id",
    "timestamp",
    "frame_id",
    "manifest_index",
    "playlist",
    "segment",
    "source_pseudo_sample_id",
    "action_label_reference_only",
    "linear",
    "angular",
    "scene_categories",
    "scene_category_source",
    "review_status",
)

PAVED_CLASSES = {"road", "sidewalk", "floor", "runway"}
OFF_ROAD_CLASSES = {"earth", "grass", "field", "sand", "path", "dirt track", "land"}
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "bus", "truck", "van", "bicycle", "minibike"}
STREET_OBJECT_CLASSES = {"bench", "pole", "streetlight", "signboard", "column"}
STRUCTURE_OBSTACLE_CLASSES = {"wall", "building", "fence", "railing", "tree"}
CURB_OR_STAIRS_CLASSES = {"stairs", "stairway", "step"}
TARGET_SCENE_CATEGORIES = (
    "PAVED_GROUND",
    "OFF_ROAD_GROUND",
    "GROUND_BOUNDARY",
    "PERSON",
    "VEHICLE",
    "STREET_FURNITURE_OR_POLE",
    "STRUCTURE_OBSTACLE",
    "CURB_OR_STAIRS",
    "TURNING",
    "SHADOW_CANDIDATE",
    "BACKLIGHT_CANDIDATE",
    "BLUR_CANDIDATE",
    "REFLECTION_CANDIDATE",
    "NARROW_PASSAGE_CANDIDATE",
)
EXPECTED_LABELS = {"IGNORE": 0, "ON_ROAD": 1, "OFF_ROAD": 2, "OBSTACLE": 3}
EXPECTED_COLORS = {
    "IGNORE": (0, 0, 0),
    "ON_ROAD": (38, 166, 91),
    "OFF_ROAD": (43, 126, 216),
    "OBSTACLE": (220, 50, 47),
    "background": (0, 0, 0),
}


@dataclass(frozen=True)
class AnnotationCandidate:
    source_sample_id: str
    image_path: Path
    pseudo_mask_path: Path
    ride_id: str
    timestamp: float
    frame_id: int
    manifest_index: int
    playlist: str
    segment: str
    action_label: str
    linear: float
    angular: float
    scene_categories: tuple[str, ...]
    category_evidence: dict[str, object]


def load_annotation_candidates(bundle_root: str | Path) -> list[AnnotationCandidate]:
    root = Path(bundle_root).expanduser().resolve()
    rows = list(csv.DictReader((root / "review.csv").open(newline="", encoding="utf-8")))
    candidates: list[AnnotationCandidate] = []
    for row in rows:
        sample_id = row["sample_id"]
        metadata = json.loads((root / "metadata" / f"{sample_id}.json").read_text(encoding="utf-8"))
        source = metadata["source"]
        image_path = _safe_file(root, row["image_path"])
        pseudo_mask_path = _safe_file(root, row["traversability_mask_path"])
        categories, evidence = classify_scene(metadata, image_path)
        candidates.append(
            AnnotationCandidate(
                source_sample_id=sample_id,
                image_path=image_path,
                pseudo_mask_path=pseudo_mask_path,
                ride_id=str(source["ride_id"]),
                timestamp=float(source["timestamp"]),
                frame_id=int(source["frame_id"]),
                manifest_index=int(source["manifest_index"]),
                playlist=str(source["playlist"]),
                segment=str(source.get("segment", "")),
                action_label=str(source["action_label_reference_only"]),
                linear=float(source["linear"]),
                angular=float(source["angular"]),
                scene_categories=categories,
                category_evidence=evidence,
            )
        )
    return candidates


def classify_scene(metadata: dict[str, object], image_path: Path) -> tuple[tuple[str, ...], dict[str, object]]:
    semantic_fractions = {
        str(item["class_name"]).strip().lower(): float(item["fraction"])
        for item in metadata.get("top_semantic_classes", [])
    }
    paved = sum(semantic_fractions.get(name, 0.0) for name in PAVED_CLASSES)
    off_road = sum(semantic_fractions.get(name, 0.0) for name in OFF_ROAD_CLASSES)
    person = sum(semantic_fractions.get(name, 0.0) for name in PERSON_CLASSES)
    vehicle = sum(semantic_fractions.get(name, 0.0) for name in VEHICLE_CLASSES)
    street_object = sum(semantic_fractions.get(name, 0.0) for name in STREET_OBJECT_CLASSES)
    structure = sum(semantic_fractions.get(name, 0.0) for name in STRUCTURE_OBSTACLE_CLASSES)
    curb_stairs = sum(semantic_fractions.get(name, 0.0) for name in CURB_OR_STAIRS_CLASSES)
    reflection = sum(semantic_fractions.get(name, 0.0) for name in {"water", "mirror", "screen"})
    source = metadata["source"]

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read candidate image: {image_path}")
    p10, p90 = (float(value) for value in np.percentile(image, (10, 90)))
    split = max(1, image.shape[0] // 2)
    top_mean = float(np.mean(image[:split]))
    bottom_mean = float(np.mean(image[split:]))
    laplacian_variance = float(cv2.Laplacian(image, cv2.CV_64F).var())

    categories: set[str] = set()
    if paved >= 0.03:
        categories.add("PAVED_GROUND")
    if off_road >= 0.03:
        categories.add("OFF_ROAD_GROUND")
    if paved >= 0.02 and off_road >= 0.02:
        categories.add("GROUND_BOUNDARY")
    if person >= 0.002:
        categories.add("PERSON")
    if vehicle >= 0.002:
        categories.add("VEHICLE")
    if street_object >= 0.002:
        categories.add("STREET_FURNITURE_OR_POLE")
    if structure >= 0.01:
        categories.add("STRUCTURE_OBSTACLE")
    if curb_stairs > 0.0:
        categories.add("CURB_OR_STAIRS")
    if str(source.get("action_label_reference_only", "")) in {"LEFT", "RIGHT"} or abs(float(source.get("angular", 0.0))) >= 0.1:
        categories.add("TURNING")
    if p10 <= 45.0 and p90 - p10 >= 120.0:
        categories.add("SHADOW_CANDIDATE")
    if top_mean - bottom_mean >= 55.0 and p90 >= 210.0:
        categories.add("BACKLIGHT_CANDIDATE")
    if 8.0 <= laplacian_variance < 80.0:
        categories.add("BLUR_CANDIDATE")
    if reflection >= 0.01:
        categories.add("REFLECTION_CANDIDATE")
    if paved + off_road >= 0.05 and structure + street_object >= 0.10:
        categories.add("NARROW_PASSAGE_CANDIDATE")
    if not categories:
        categories.add("OTHER")
    evidence = {
        "semantic_fraction": {
            "paved": paved,
            "off_road": off_road,
            "person": person,
            "vehicle": vehicle,
            "street_furniture_or_pole": street_object,
            "structure_obstacle": structure,
            "curb_or_stairs": curb_stairs,
            "reflection": reflection,
        },
        "pseudo_unknown_fraction": float(
            metadata.get("traversability_class_area", {})
            .get("UNKNOWN_OR_IGNORE", {})
            .get("fraction", 0.0)
        ),
        "top_semantic_fraction": semantic_fractions,
        "image_luminance": {
            "p10": p10,
            "p90": p90,
            "top_mean": top_mean,
            "bottom_mean": bottom_mean,
            "laplacian_variance": laplacian_variance,
        },
        "warning": "Categories are deterministic candidates for sampling and require human confirmation.",
    }
    return tuple(sorted(categories)), evidence


def select_annotation_candidates(
    candidates: list[AnnotationCandidate],
    requested_count: int,
    minimum_separation_seconds: float,
    seed: int,
) -> list[AnnotationCandidate]:
    if requested_count <= 0 or minimum_separation_seconds < 0:
        raise ValueError("requested count must be positive and separation cannot be negative")
    remaining = list(candidates)
    selected: list[AnnotationCandidate] = []
    covered: set[str] = set()
    selected_rides: Counter[str] = Counter()
    while remaining and len(selected) < requested_count:
        eligible = [
            candidate
            for candidate in remaining
            if all(
                candidate.ride_id != chosen.ride_id
                or abs(candidate.timestamp - chosen.timestamp) >= minimum_separation_seconds - 1e-9
                for chosen in selected
            )
        ]
        if not eligible:
            break
        eligible.sort(
            key=lambda candidate: (
                -100 * len(set(candidate.scene_categories) - covered)
                - 40 * int(selected_rides[candidate.ride_id] == 0)
                + 5 * selected_rides[candidate.ride_id],
                _stable_key(candidate, seed),
            )
        )
        chosen = eligible[0]
        selected.append(chosen)
        covered.update(chosen.scene_categories)
        selected_rides[chosen.ride_id] += 1
        remaining.remove(chosen)
    return selected


def build_annotation_bundle(
    selected: list[AnnotationCandidate],
    output_dir: str | Path,
    contract_path: str | Path,
    source_bundle: str | Path,
    seed: int,
    minimum_separation_seconds: float,
    sample_id_prefix: str = "trav_v1_",
) -> dict[str, object]:
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"output directory is not empty: {root}")
    for name in ("images", "masks", "metadata", "initial_masks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(contract_path, root / "label_contract.yaml")
    name_to_id, colors, _ = label_contract(root / "label_contract.yaml")
    _require_v1_contract(name_to_id)
    _write_labelmap(name_to_id, colors, root / "cvat_labelmap.txt")

    entries: list[dict[str, str]] = []
    for position, candidate in enumerate(selected):
        sample_id = f"{sample_id_prefix}{position:05d}"
        image_suffix = candidate.image_path.suffix.lower() or ".jpg"
        image_relative = f"images/{sample_id}{image_suffix}"
        shutil.copy2(candidate.image_path, root / image_relative)
        initial_relative = f"initial_masks/{sample_id}.png"
        convert_pseudo_seed(candidate.pseudo_mask_path, root / initial_relative)
        entry = {
            "sample_id": sample_id,
            "image_path": image_relative,
            "mask_path": f"masks/{sample_id}.png",
            "ride_id": candidate.ride_id,
            "timestamp": f"{candidate.timestamp:.6f}",
            "frame_id": str(candidate.frame_id),
            "manifest_index": str(candidate.manifest_index),
            "playlist": candidate.playlist,
            "segment": candidate.segment,
            "source_pseudo_sample_id": candidate.source_sample_id,
            "action_label_reference_only": candidate.action_label,
            "linear": f"{candidate.linear:.6f}",
            "angular": f"{candidate.angular:.6f}",
            "scene_categories": "|".join(candidate.scene_categories),
            "scene_category_source": "semantic_pseudo_label_and_image_heuristics_unverified",
            "review_status": "NOT_ANNOTATED",
        }
        entries.append(entry)
        write_json(
            root / "metadata" / f"{sample_id}.json",
            {
                **entry,
                "scene_category_evidence": candidate.category_evidence,
                "initial_mask_path": initial_relative,
                "initial_mask_mapping": {
                    "old_0_NON_TRAVERSABLE": "3_OBSTACLE",
                    "old_1_TRAVERSABLE": "1_ON_ROAD",
                    "old_2_UNKNOWN_OR_IGNORE": "0_IGNORE",
                    "OFF_ROAD": "never_auto_seeded",
                },
                "pseudo_label_is_ground_truth": False,
            },
        )
    _write_metadata_csv(entries, root / "metadata.csv")
    write_contact_sheet(entries, root, root / "contact_sheet.jpg")
    write_cvat_seed_archive(entries, root, root / "cvat_seed_annotations.zip")
    _write_bundle_readme(root, len(entries))
    category_counts = Counter(
        category
        for entry in entries
        for category in entry["scene_categories"].split("|")
    )
    report = {
        "pipeline_status": "HUMAN_ANNOTATION_REQUIRED",
        "dataset_name": "traversability_dataset_v1",
        "source_pseudo_bundle": str(Path(source_bundle).expanduser().resolve()),
        "source_candidate_count": _count_csv_rows(Path(source_bundle).expanduser().resolve() / "review.csv"),
        "selected_sample_count": len(entries),
        "processed_ride_count": len({entry["ride_id"] for entry in entries}),
        "ride_distribution": dict(sorted(Counter(entry["ride_id"] for entry in entries).items())),
        "scene_category_distribution": dict(sorted(category_counts.items())),
        "target_categories_not_found": sorted(set(TARGET_SCENE_CATEGORIES) - set(category_counts)),
        "minimum_separation_seconds": minimum_separation_seconds,
        "seed": seed,
        "pseudo_label_inference_performed": False,
        "raw_dataset_accessed": False,
        "model_training_performed": False,
        "live_rover_commands_sent": False,
        "annotation_tool": "CVAT Segmentation Mask 1.1",
        "initial_masks_are_unverified": True,
    }
    write_json(root / "build_report.json", report)
    return report


def convert_pseudo_seed(source: Path, destination: Path) -> None:
    mask = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise ValueError(f"pseudo mask must be a single-channel PNG: {source}")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 1, 2}):
        raise ValueError(f"pseudo mask contains unsupported IDs: {sorted(values)}")
    converted = np.zeros(mask.shape, dtype=np.uint8)
    converted[mask == 0] = 3
    converted[mask == 1] = 1
    if not cv2.imwrite(str(destination), converted):
        raise OSError(f"cannot write initial mask: {destination}")


def import_cvat_masks(
    bundle_root: str | Path,
    cvat_export: str | Path,
    output_dir: str | Path,
    expected_count: int | None = None,
) -> dict[str, object]:
    root = Path(bundle_root).expanduser().resolve()
    export = Path(cvat_export).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.parent != root:
        raise ValueError("output directory must be a direct child of the pilot bundle")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    preflight = validate_annotation_dataset(root, require_masks=False)
    if not preflight["valid"]:
        raise ValueError(f"annotation bundle preflight failed: {preflight['errors']}")
    rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
    name_to_id, colors, _ = label_contract(root / "label_contract.yaml")
    _require_v1_contract(name_to_id)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} metadata rows, found {len(rows)}")
    expected = {row["sample_id"]: row for row in rows}
    label_entries, found = read_cvat_segmentation_export(export)
    index_to_id, color_to_id = cvat_label_mapping(label_entries)
    if expected_count is not None and len(found) != expected_count:
        raise ValueError(f"expected {expected_count} SegmentationClass masks, found {len(found)}")
    missing = sorted(set(expected) - set(found))
    extras = sorted(set(found) - set(expected))
    if missing or extras:
        raise ValueError(f"CVAT mask set mismatch; missing={missing}, extras={extras}")

    converted: dict[str, np.ndarray] = {}
    for sample_id, row in expected.items():
        mask = normalize_cvat_class_mask(found[sample_id], index_to_id, color_to_id)
        image = cv2.imread(str(root / row["image_path"]), cv2.IMREAD_COLOR)
        if image is None or mask.shape != image.shape[:2]:
            raise ValueError(f"{sample_id}: mask dimensions do not match the source image")
        converted[sample_id] = mask

    masks_dir = output / "masks"
    overlays_dir = output / "overlays"
    mask_visualizations_dir = output / "mask_visualizations"
    masks_dir.mkdir(parents=True, exist_ok=False)
    overlays_dir.mkdir(parents=True, exist_ok=False)
    mask_visualizations_dir.mkdir(parents=True, exist_ok=False)
    for sample_id, mask in converted.items():
        if not cv2.imwrite(str(masks_dir / f"{sample_id}.png"), mask):
            raise OSError(f"cannot write imported mask: {sample_id}")
        image = cv2.imread(str(root / expected[sample_id]["image_path"]), cv2.IMREAD_COLOR)
        assert image is not None
        color = colorize_mask(mask, colors)
        if not cv2.imwrite(
            str(mask_visualizations_dir / f"{sample_id}.png"),
            cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
        ):
            raise OSError(f"cannot write mask visualization: {sample_id}")
        overlay = cv2.addWeighted(image, 0.55, cv2.cvtColor(color, cv2.COLOR_RGB2BGR), 0.45, 0.0)
        if not cv2.imwrite(str(overlays_dir / f"{sample_id}.jpg"), overlay):
            raise OSError(f"cannot write overlay: {sample_id}")
    report = {
        "imported_mask_count": len(converted),
        "source": str(export),
        "output_dir": str(output),
        "masks_dir": str(masks_dir),
        "labelmap_entries": [
            {"index": index, "name": name, "color_rgb": list(color), "final_class_id": index_to_id[index]}
            for index, (name, color) in enumerate(label_entries)
        ],
        "semantic_mask_source": "SegmentationClass",
        "segmentation_object_used": False,
        "background_merged_into_ignore": any(name == "background" for name, _ in label_entries),
    }
    write_json(output / "import_report.json", report)
    return report


def read_cvat_segmentation_export(
    export: Path,
) -> tuple[list[tuple[str, tuple[int, int, int]]], dict[str, np.ndarray]]:
    found: dict[str, np.ndarray] = {}
    labelmap_texts: list[str] = []
    if export.is_dir():
        for path in export.rglob("*.png"):
            if "SegmentationClass" in path.parts:
                if path.stem in found:
                    raise ValueError(f"duplicate CVAT mask stem: {path.stem}")
                found[path.stem] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        labelmap_paths = list(export.rglob("labelmap.txt"))
        labelmap_texts = [path.read_text(encoding="utf-8") for path in labelmap_paths]
    elif export.is_file() and zipfile.is_zipfile(export):
        with zipfile.ZipFile(export) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"CVAT ZIP contains a corrupt member: {bad}")
            for name in archive.namelist():
                member = PurePosixPath(name)
                if member.suffix.lower() == ".png" and "SegmentationClass" in member.parts:
                    if member.stem in found:
                        raise ValueError(f"duplicate CVAT mask stem: {member.stem}")
                    encoded = np.frombuffer(archive.read(name), dtype=np.uint8)
                    found[member.stem] = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                elif member.name == "labelmap.txt":
                    labelmap_texts.append(archive.read(name).decode("utf-8"))
    else:
        raise ValueError("CVAT export must be a directory or ZIP archive")
    if len(labelmap_texts) != 1:
        raise ValueError(f"expected exactly one labelmap.txt, found {len(labelmap_texts)}")
    return parse_cvat_labelmap(labelmap_texts[0]), found


def parse_cvat_labelmap(text: str) -> list[tuple[str, tuple[int, int, int]]]:
    entries: list[tuple[str, tuple[int, int, int]]] = []
    seen_names: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            raise ValueError(f"invalid labelmap line {line_number}: {raw_line!r}")
        name = parts[0].strip()
        if name not in EXPECTED_COLORS:
            raise ValueError(f"unknown CVAT label: {name!r}")
        if name in seen_names:
            raise ValueError(f"duplicate CVAT label: {name!r}")
        seen_names.add(name)
        try:
            color = tuple(int(channel.strip()) for channel in parts[1].split(","))
        except ValueError as exc:
            raise ValueError(f"invalid RGB value for CVAT label {name!r}") from exc
        if len(color) != 3 or color != EXPECTED_COLORS[name]:
            raise ValueError(
                f"CVAT label {name!r} uses RGB {color}, expected {EXPECTED_COLORS[name]}"
            )
        entries.append((name, color))
    missing = set(EXPECTED_LABELS) - seen_names
    if missing:
        raise ValueError(f"CVAT labelmap is missing required labels: {sorted(missing)}")
    return entries


def cvat_label_mapping(
    entries: list[tuple[str, tuple[int, int, int]]],
) -> tuple[dict[int, int], dict[tuple[int, int, int], int]]:
    index_to_id: dict[int, int] = {}
    color_to_id: dict[tuple[int, int, int], int] = {}
    for index, (name, color) in enumerate(entries):
        final_id = 0 if name == "background" else EXPECTED_LABELS[name]
        index_to_id[index] = final_id
        existing = color_to_id.get(color)
        if existing is not None and existing != final_id:
            raise ValueError(f"RGB {color} maps to conflicting final class IDs")
        color_to_id[color] = final_id
    return index_to_id, color_to_id


def normalize_cvat_class_mask(
    mask: np.ndarray | None,
    index_to_id: dict[int, int],
    color_to_id: dict[tuple[int, int, int], int],
) -> np.ndarray:
    if mask is None:
        raise ValueError("SegmentationClass mask is unreadable")
    result = np.full(mask.shape[:2], 255, dtype=np.uint8)
    if mask.ndim == 2:
        values = set(int(value) for value in np.unique(mask))
        unknown = values - set(index_to_id)
        if unknown:
            raise ValueError(f"SegmentationClass mask uses unknown labelmap indices: {sorted(unknown)}")
        for source_index, final_id in index_to_id.items():
            result[mask == source_index] = final_id
    elif mask.ndim == 3 and mask.shape[2] in {3, 4}:
        rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
        for color, final_id in color_to_id.items():
            result[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = final_id
    else:
        raise ValueError("SegmentationClass mask must be a 1-channel or RGB PNG")
    if np.any(result == 255):
        raise ValueError("SegmentationClass mask contains an unknown index or RGB color")
    return result


def validate_annotation_dataset(
    bundle_root: str | Path,
    require_masks: bool = True,
    masks_dir: str | Path | None = None,
) -> dict[str, object]:
    root = Path(bundle_root).expanduser().resolve()
    resolved_masks_dir = (
        Path(masks_dir).expanduser().resolve()
        if masks_dir is not None
        else (root / "masks").resolve()
    )
    rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
    name_to_id, _, _ = label_contract(root / "label_contract.yaml")
    _require_v1_contract(name_to_id)
    allowed_ids = set(name_to_id.values())
    errors: list[str] = []
    seen: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    expected_masks: set[Path] = set()
    pixel_counts: Counter[int] = Counter()
    per_image: dict[str, dict[str, object]] = {}
    all_ignore: list[str] = []
    single_class: list[str] = []
    image_references: set[Path] = set()
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in seen:
            errors.append(f"missing or duplicate sample_id: {sample_id!r}")
            continue
        seen.add(sample_id)
        try:
            for field in ("ride_id", "timestamp", "frame_id", "manifest_index", "playlist", "segment"):
                if not row.get(field, "").strip():
                    raise ValueError(f"required metadata field is empty: {field}")
            float(row["timestamp"])
            int(row["frame_id"])
            int(row["manifest_index"])
            source_key = (row["ride_id"], row["manifest_index"])
            if source_key in seen_sources:
                raise ValueError(f"duplicate source frame metadata: {source_key}")
            seen_sources.add(source_key)
            image_path = _safe_file(root, row["image_path"])
            if image_path.stem != sample_id:
                raise ValueError("image filename does not match sample_id")
            if image_path in image_references:
                raise ValueError("duplicate image reference")
            image_references.add(image_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("source image is unreadable")
            metadata = json.loads((root / "metadata" / f"{sample_id}.json").read_text(encoding="utf-8"))
            if metadata.get("sample_id") != sample_id:
                raise ValueError("JSON metadata sample_id differs")
            declared_mask = Path(row["mask_path"])
            mask_path = resolved_masks_dir / f"{sample_id}.png"
            if (
                declared_mask.stem != sample_id
                or declared_mask.suffix.lower() != ".png"
                or declared_mask.parent != Path("masks")
            ):
                raise ValueError("declared mask filename or directory does not match sample_id")
            expected_masks.add(mask_path)
            if not mask_path.is_file():
                if require_masks:
                    raise ValueError("mask file is missing")
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                raise ValueError("mask must be a readable single-channel PNG")
            if mask.dtype != np.uint8:
                raise ValueError("mask must use 8-bit class IDs")
            if mask.shape != image.shape[:2]:
                raise ValueError("image and mask dimensions differ")
            values, counts = np.unique(mask, return_counts=True)
            invalid = set(int(value) for value in values) - allowed_ids
            if invalid:
                raise ValueError(f"mask contains invalid class IDs: {sorted(invalid)}")
            image_counts = {int(value): int(count) for value, count in zip(values, counts)}
            pixel_counts.update(image_counts)
            total = int(mask.size)
            present_ids = sorted(image_counts)
            if present_ids == [0]:
                all_ignore.append(sample_id)
            if len(present_ids) == 1:
                single_class.append(sample_id)
            per_image[sample_id] = {
                "ride_id": row["ride_id"],
                "timestamp": float(row["timestamp"]),
                "frame_id": int(row["frame_id"]),
                "manifest_index": int(row["manifest_index"]),
                "class_pixels": {
                    name: image_counts.get(class_id, 0)
                    for name, class_id in sorted(name_to_id.items(), key=lambda item: item[1])
                },
                "class_fractions": {
                    name: image_counts.get(class_id, 0) / total
                    for name, class_id in sorted(name_to_id.items(), key=lambda item: item[1])
                },
                "single_class": len(present_ids) == 1,
                "all_ignore": present_ids == [0],
            }
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{sample_id}: {exc}")
    actual_masks = set(resolved_masks_dir.glob("*.png")) if resolved_masks_dir.is_dir() else set()
    extra_masks = actual_masks - expected_masks
    if extra_masks:
        errors.append(f"unexpected mask files: {sorted(path.name for path in extra_masks)}")
    if require_masks and not errors:
        if pixel_counts[EXPECTED_LABELS["ON_ROAD"]] == 0:
            errors.append("ON_ROAD is absent from all masks")
        if pixel_counts[EXPECTED_LABELS["OFF_ROAD"]] == 0:
            errors.append("OFF_ROAD is absent from all masks")
    total_pixels = sum(pixel_counts.values())
    class_pixel_counts = {
        name: pixel_counts[class_id]
        for name, class_id in sorted(name_to_id.items(), key=lambda item: item[1])
    }
    return {
        "valid": not errors,
        "sample_count": len(rows),
        "validated_mask_count": sum(1 for path in expected_masks if path.is_file()),
        "require_masks": require_masks,
        "masks_dir": str(resolved_masks_dir),
        "class_pixel_counts": class_pixel_counts,
        "class_pixel_fractions": {
            name: count / total_pixels if total_pixels else 0.0
            for name, count in class_pixel_counts.items()
        },
        "per_image": per_image,
        "empty_mask_sample_ids": [],
        "all_ignore_sample_ids": all_ignore,
        "single_class_sample_ids": single_class,
        "warnings": [
            *([f"all-IGNORE masks: {all_ignore}"] if all_ignore else []),
            *([f"single-class masks: {single_class}"] if single_class else []),
        ],
        "errors": errors,
    }


def write_annotation_review_outputs(
    bundle_root: str | Path,
    output_dir: str | Path,
    validation: dict[str, object],
) -> None:
    root = Path(bundle_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    per_image = validation["per_image"]
    if not isinstance(per_image, dict) or not per_image:
        raise ValueError("cannot create review outputs without validated image statistics")
    rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
    fields = [
        "sample_id", "ride_id", "timestamp", "frame_id", "manifest_index",
        "IGNORE_pixels", "IGNORE_fraction", "ON_ROAD_pixels", "ON_ROAD_fraction",
        "OFF_ROAD_pixels", "OFF_ROAD_fraction", "OBSTACLE_pixels", "OBSTACLE_fraction",
        "single_class", "all_ignore",
    ]
    with (output / "per_image_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stats = per_image[row["sample_id"]]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "ride_id": stats["ride_id"],
                    "timestamp": stats["timestamp"],
                    "frame_id": stats["frame_id"],
                    "manifest_index": stats["manifest_index"],
                    **{
                        f"{name}_pixels": stats["class_pixels"][name]
                        for name in EXPECTED_LABELS
                    },
                    **{
                        f"{name}_fraction": stats["class_fractions"][name]
                        for name in EXPECTED_LABELS
                    },
                    "single_class": stats["single_class"],
                    "all_ignore": stats["all_ignore"],
                }
            )
    _write_overlay_contact_sheet(rows, root, output)
    _write_annotation_review_html(rows, output, per_image)


def _write_overlay_contact_sheet(rows: list[dict[str, str]], root: Path, output: Path) -> None:
    tile_width, tile_height, caption_height = 320, 180, 32
    columns = 3
    canvas = np.full(
        (len(rows) * (tile_height + caption_height), columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )
    for row_index, row in enumerate(rows):
        sample_id = row["sample_id"]
        image = cv2.imread(str(root / row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(output / "masks" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        overlay = cv2.imread(str(output / "overlays" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
        if image is None or mask is None or overlay is None:
            raise OSError(f"cannot read review assets for {sample_id}")
        mask_rgb = colorize_mask(mask, {class_id: EXPECTED_COLORS[name] for name, class_id in EXPECTED_LABELS.items()})
        assets = (image, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR), overlay)
        interpolations = (cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_AREA)
        y = row_index * (tile_height + caption_height)
        for column, (asset, interpolation) in enumerate(zip(assets, interpolations)):
            x = column * tile_width
            canvas[y : y + tile_height, x : x + tile_width] = _fit_bgr(
                asset,
                tile_width,
                tile_height,
                interpolation,
            )
        caption = f"{sample_id} ride={row['ride_id']} frame={row['frame_id']} | original / mask / overlay"
        cv2.putText(
            canvas,
            caption,
            (6, y + tile_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output / "overlay_contact_sheet.jpg"), canvas):
        raise OSError("cannot write overlay contact sheet")


def _write_annotation_review_html(
    rows: list[dict[str, str]],
    output: Path,
    per_image: dict[str, object],
) -> None:
    cards: list[str] = []
    for row in rows:
        sample_id = row["sample_id"]
        stats = per_image[sample_id]
        cards.append(
            "<article><h2>{sample}</h2><p>ride={ride} timestamp={timestamp} frame={frame}</p>"
            "<div class='grid'><figure><img src='../{image}'><figcaption>Original</figcaption></figure>"
            "<figure><img class='pixel' src='mask_visualizations/{sample}.png'><figcaption>Colorized class-ID mask</figcaption></figure>"
            "<figure><img src='overlays/{sample}.jpg'><figcaption>Overlay</figcaption></figure></div>"
            "<pre>{stats}</pre></article>".format(
                sample=html.escape(sample_id),
                ride=html.escape(row["ride_id"]),
                timestamp=html.escape(row["timestamp"]),
                frame=html.escape(row["frame_id"]),
                image=html.escape(row["image_path"]),
                stats=html.escape(json.dumps(stats, indent=2, sort_keys=True)),
            )
        )
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Reviewed Traversability Masks</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;background:#f4f5f6;color:#17191c}article{background:white;border:1px solid #ccd0d5;margin:0 0 24px;padding:16px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto;display:block}.pixel{image-rendering:pixelated}figcaption{padding-top:6px;font-weight:600}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
</head><body><h1>Reviewed Traversability Masks</h1><p>Legend: black IGNORE, green ON_ROAD, blue OFF_ROAD, red OBSTACLE.</p>__CARDS__</body></html>""".replace("__CARDS__", "\n".join(cards))
    (output / "review.html").write_text(document, encoding="utf-8")


def _fit_bgr(image: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def class_id_mask(mask: np.ndarray | None, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    if mask is None:
        raise ValueError("mask is unreadable")
    if mask.ndim == 2:
        values = set(int(value) for value in np.unique(mask))
        invalid = values - set(colors)
        if invalid:
            raise ValueError(f"mask contains unsupported IDs: {sorted(invalid)}")
        result = mask.astype(np.uint8, copy=False)
    elif mask.ndim == 3 and mask.shape[2] in {3, 4}:
        rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
        result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for class_id, color in colors.items():
            result[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = class_id
    else:
        raise ValueError("mask must be single-channel IDs or exact contract RGB colors")
    invalid = set(int(value) for value in np.unique(result)) - set(colors)
    if invalid:
        raise ValueError(f"mask contains unsupported IDs or colors: {sorted(invalid)}")
    return result


def write_cvat_seed_archive(entries: list[dict[str, str]], root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / "cvat_labelmap.txt", "labelmap.txt")
        image_set = "\n".join(entry["sample_id"] for entry in entries) + "\n"
        archive.writestr("ImageSets/Segmentation/default.txt", image_set)
        for entry in entries:
            sample_id = entry["sample_id"]
            mask_path = root / "initial_masks" / f"{sample_id}.png"
            archive.write(mask_path, f"SegmentationClass/{sample_id}.png")
            archive.write(mask_path, f"SegmentationObject/{sample_id}.png")


def _write_labelmap(name_to_id: dict[str, int], colors: dict[int, tuple[int, int, int]], output: Path) -> None:
    id_to_name = {class_id: name for name, class_id in name_to_id.items()}
    lines = ["# label : color (RGB) : body parts : actions"]
    for class_id in sorted(id_to_name):
        red, green, blue = colors[class_id]
        lines.append(f"{id_to_name[class_id]}:{red},{green},{blue}::")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _require_v1_contract(name_to_id: dict[str, int]) -> None:
    if name_to_id != EXPECTED_LABELS:
        raise ValueError(f"traversability_dataset_v1 requires exactly {EXPECTED_LABELS}")


def _write_metadata_csv(entries: list[dict[str, str]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(entries)


def _write_bundle_readme(root: Path, sample_count: int) -> None:
    text = f"""# Traversability Dataset v1 Annotation Bundle

This {sample_count}-image bundle is for manual pixel annotation. It is not a training dataset until every mask passes the Dell validator and the user approves it.

## Labels

- `0 IGNORE`: uncertain, sky, rover hood, severe shadow/reflection, or excluded pixels.
- `1 ON_ROAD`: traversable paved ground.
- `2 OFF_ROAD`: traversable unpaved ground.
- `3 OBSTACLE`: people, vehicles, furniture, poles, walls, curbs, stairs, and other no-entry regions.

Paved preference belongs in a later planner cost. Do not change a traversable unpaved pixel to `OBSTACLE` merely to prefer pavement.

## CVAT workflow on Mac

CVAT is used because its standard `Segmentation Mask 1.1` format directly exports pixel masks and a label map; Label Studio would require additional JSON/RLE conversion. Create a CVAT task with the files in `images/` and labels from `cvat_labelmap.txt` in exact ID order. Annotate from scratch, or import `cvat_seed_annotations.zip` as `Segmentation Mask 1.1` after the images are attached.

The seed masks are unverified conversions of the old 3-class pseudo-labels: old non-traversable becomes `OBSTACLE`, old traversable becomes `ON_ROAD`, and old unknown becomes `IGNORE`. They never seed `OFF_ROAD` and must be corrected completely. Scene categories in metadata are also unverified sampling hints.

Export annotations from CVAT as `Segmentation Mask 1.1` without source images. Return the ZIP to Dell. On Dell, import it with `training/import_cvat_traversability_masks.py`, then run `training/validate_traversability_dataset_v1.py`. Do not train or expand the dataset before human review and explicit approval.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes bundle: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"file is missing: {relative}")
    return path


def _stable_key(candidate: AnnotationCandidate, seed: int) -> str:
    value = f"{seed}:{candidate.ride_id}:{candidate.timestamp:.9f}:{candidate.frame_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))
