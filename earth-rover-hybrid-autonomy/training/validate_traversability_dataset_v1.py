#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.traversability_annotation import validate_annotation_dataset
from training.traversability_review import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate completed traversability_dataset_v1 image-mask pairs.")
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    root = Path(args.bundle).expanduser().resolve()
    report = validate_annotation_dataset(root, require_masks=True)
    write_json(root / "validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
