from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from training.traversability_review import label_contract, write_contact_sheet, write_json


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
STRUCTURE_OBSTACLE_CLASSES = {"wall", "building", "fence", "railing"}
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
)
EXPECTED_LABELS = {"IGNORE": 0, "ON_ROAD": 1, "OFF_ROAD": 2, "OBSTACLE": 3}


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
    source = metadata["source"]

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read candidate image: {image_path}")
    p10, p90 = (float(value) for value in np.percentile(image, (10, 90)))
    split = max(1, image.shape[0] // 2)
    top_mean = float(np.mean(image[:split]))
    bottom_mean = float(np.mean(image[split:]))

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
        },
        "top_semantic_fraction": semantic_fractions,
        "image_luminance": {
            "p10": p10,
            "p90": p90,
            "top_mean": top_mean,
            "bottom_mean": bottom_mean,
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
        sample_id = f"trav_v1_{position:05d}"
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
    _write_bundle_readme(root)
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


def import_cvat_masks(bundle_root: str | Path, cvat_export: str | Path) -> dict[str, object]:
    root = Path(bundle_root).expanduser().resolve()
    export = Path(cvat_export).expanduser().resolve()
    preflight = validate_annotation_dataset(root, require_masks=False)
    if not preflight["valid"]:
        raise ValueError(f"annotation bundle preflight failed: {preflight['errors']}")
    rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
    name_to_id, colors, _ = label_contract(root / "label_contract.yaml")
    _require_v1_contract(name_to_id)
    expected = {row["sample_id"]: row for row in rows}
    found: dict[str, np.ndarray] = {}
    if export.is_dir():
        for path in export.rglob("*.png"):
            if "SegmentationClass" in path.parts:
                if path.stem in found:
                    raise ValueError(f"duplicate CVAT mask stem: {path.stem}")
                found[path.stem] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    elif export.is_file() and zipfile.is_zipfile(export):
        with zipfile.ZipFile(export) as archive:
            for name in archive.namelist():
                member = PurePosixPath(name)
                if member.suffix.lower() == ".png" and "SegmentationClass" in member.parts:
                    if member.stem in found:
                        raise ValueError(f"duplicate CVAT mask stem: {member.stem}")
                    encoded = np.frombuffer(archive.read(name), dtype=np.uint8)
                    found[member.stem] = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError("CVAT export must be a directory or ZIP archive")
    missing = sorted(set(expected) - set(found))
    extras = sorted(set(found) - set(expected))
    if missing or extras:
        raise ValueError(f"CVAT mask set mismatch; missing={missing}, extras={extras}")
    masks_dir = root / "masks"
    if any(masks_dir.glob("*.png")):
        raise ValueError("masks directory is not empty; preserve the existing import")
    converted: dict[str, np.ndarray] = {}
    for sample_id, row in expected.items():
        mask = class_id_mask(found[sample_id], colors)
        image = cv2.imread(str(root / row["image_path"]), cv2.IMREAD_COLOR)
        if image is None or mask.shape != image.shape[:2]:
            raise ValueError(f"{sample_id}: mask dimensions do not match the source image")
        converted[sample_id] = mask
    for sample_id, mask in converted.items():
        if not cv2.imwrite(str(masks_dir / f"{sample_id}.png"), mask):
            raise OSError(f"cannot write imported mask: {sample_id}")
    return {"imported_mask_count": len(expected), "source": str(export), "output_dir": str(masks_dir)}


def validate_annotation_dataset(bundle_root: str | Path, require_masks: bool = True) -> dict[str, object]:
    root = Path(bundle_root).expanduser().resolve()
    rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
    name_to_id, _, _ = label_contract(root / "label_contract.yaml")
    _require_v1_contract(name_to_id)
    allowed_ids = set(name_to_id.values())
    errors: list[str] = []
    seen: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    expected_masks: set[Path] = set()
    pixel_counts: Counter[int] = Counter()
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
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("source image is unreadable")
            metadata = json.loads((root / "metadata" / f"{sample_id}.json").read_text(encoding="utf-8"))
            if metadata.get("sample_id") != sample_id:
                raise ValueError("JSON metadata sample_id differs")
            mask_path = (root / row["mask_path"]).resolve()
            if (
                mask_path.stem != sample_id
                or mask_path.suffix.lower() != ".png"
                or mask_path.parent != (root / "masks").resolve()
            ):
                raise ValueError("mask filename or directory does not match sample_id")
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
            pixel_counts.update({int(value): int(count) for value, count in zip(values, counts)})
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{sample_id}: {exc}")
    actual_masks = set((root / "masks").glob("*.png"))
    extras = sorted(str(path.relative_to(root)) for path in actual_masks - expected_masks)
    if extras:
        errors.append(f"unexpected mask files: {extras}")
    return {
        "valid": not errors,
        "sample_count": len(rows),
        "validated_mask_count": sum(1 for path in expected_masks if path.is_file()),
        "require_masks": require_masks,
        "class_pixel_counts": {
            name: pixel_counts[class_id]
            for name, class_id in sorted(name_to_id.items(), key=lambda item: item[1])
        },
        "errors": errors,
    }


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


def _write_bundle_readme(root: Path) -> None:
    text = """# Traversability Dataset v1 Annotation Pilot

This 20-image bundle is for manual pixel annotation. It is not a training dataset until every mask passes the Dell validator and the user approves expansion.

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
