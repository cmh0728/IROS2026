#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.traversability_annotation import import_cvat_masks, validate_annotation_dataset
from training.traversability_review import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a CVAT Segmentation Mask export into traversability_dataset_v1.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--cvat-export", required=True)
    args = parser.parse_args()
    report = import_cvat_masks(args.bundle, args.cvat_export)
    validation = validate_annotation_dataset(args.bundle, require_masks=True)
    report["validation"] = validation
    write_json(Path(args.bundle).expanduser().resolve() / "import_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
