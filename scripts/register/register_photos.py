"""
Register and validate local photos for webcam testing.

This utility prepares images in data/photos for local testing mode in main.py.
It validates image readability, groups identities, and writes a JSON report.

Supported layouts:
1) data/photos/<identity_name>/*.jpg
2) data/photos/*.jpg (identity inferred from file stem or prefix)

Usage:
    python scripts/register/register_photos.py
    python scripts/register/register_photos.py --photos-dir data/photos --min-images 2
    python scripts/register/register_photos.py --organize-flat
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PhotoRecord:
    path: str
    identity: str
    valid: bool
    reason: str


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def infer_identity_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    base = stem.replace("-", "_").replace(" ", "_")
    parts = [p for p in base.split("_") if p]
    if not parts:
        return stem or "unknown"

    stop_tokens = {
        "full", "face", "img", "image", "photo", "pic",
        "selfie", "front", "profile", "left", "right",
    }
    name_parts: List[str] = []
    for token in parts:
        t = token.lower()
        if t in stop_tokens or t.isdigit():
            break
        name_parts.append(token)

    if not name_parts:
        name_parts = [parts[0]]
    return "_".join(name_parts)


def validate_image(path: Path) -> Tuple[bool, str]:
    img = cv2.imread(str(path))
    if img is None:
        return False, "unreadable"
    if img.size == 0:
        return False, "empty"
    return True, "ok"


def collect_photos(photos_dir: Path) -> List[PhotoRecord]:
    records: List[PhotoRecord] = []

    if not photos_dir.exists():
        return records

    for item in sorted(photos_dir.iterdir()):
        if item.is_dir():
            identity = item.name.strip()
            for img in sorted(item.iterdir()):
                if not is_image_file(img):
                    continue
                valid, reason = validate_image(img)
                records.append(PhotoRecord(str(img), identity, valid, reason))
        elif is_image_file(item):
            identity = infer_identity_from_filename(item)
            valid, reason = validate_image(item)
            records.append(PhotoRecord(str(item), identity, valid, reason))

    return records


def organize_flat_files(photos_dir: Path, dry_run: bool = False) -> int:
    moved = 0
    for item in sorted(photos_dir.iterdir()):
        if not is_image_file(item):
            continue
        identity = infer_identity_from_filename(item)
        target_dir = photos_dir / identity
        target_path = target_dir / item.name
        if target_path == item:
            continue

        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(target_path))
        moved += 1
    return moved


def build_report(records: List[PhotoRecord], min_images: int) -> Dict:
    valid = [r for r in records if r.valid]
    invalid = [r for r in records if not r.valid]

    identity_to_images: Dict[str, List[str]] = {}
    for rec in valid:
        identity_to_images.setdefault(rec.identity, []).append(rec.path)

    eligible = {
        identity: sorted(paths)
        for identity, paths in identity_to_images.items()
        if len(paths) >= min_images
    }

    report = {
        "total_images": len(records),
        "valid_images": len(valid),
        "invalid_images": len(invalid),
        "identities_found": len(identity_to_images),
        "eligible_identities": len(eligible),
        "min_images_required": min_images,
        "eligible_identity_map": eligible,
        "invalid_records": [asdict(r) for r in invalid],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and register local photos for webcam testing"
    )
    parser.add_argument("--photos-dir", default="data/photos", help="Path to local photos directory")
    parser.add_argument("--min-images", type=int, default=1, help="Minimum valid images per identity")
    parser.add_argument(
        "--report-path",
        default="data/photos/registration_report.json",
        help="Path to write registration report JSON",
    )
    parser.add_argument(
        "--organize-flat",
        action="store_true",
        help="Move flat files into identity folders inferred from filename",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without moving files",
    )
    args = parser.parse_args()

    photos_dir = Path(args.photos_dir)
    if not photos_dir.exists():
        print(f"❌ Photos folder not found: {photos_dir}")
        return

    if args.organize_flat:
        moved = organize_flat_files(photos_dir, dry_run=args.dry_run)
        mode = "previewed" if args.dry_run else "moved"
        print(f"ℹ️ Flat files {mode}: {moved}")

    records = collect_photos(photos_dir)
    report = build_report(records, min_images=max(args.min_images, 1))

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print("\n✅ Local photo registration scan complete")
    print(f"   Photos folder      : {photos_dir}")
    print(f"   Total images       : {report['total_images']}")
    print(f"   Valid images       : {report['valid_images']}")
    print(f"   Invalid images     : {report['invalid_images']}")
    print(f"   Identities found   : {report['identities_found']}")
    print(f"   Eligible identities: {report['eligible_identities']}")
    print(f"   Report saved       : {report_path}")

    if report["eligible_identities"] > 0:
        print("\nNext step:")
        print("  python main.py --local --all-datasets")
    else:
        print("\nNo eligible identities found. Add images under data/photos and try again.")


if __name__ == "__main__":
    main()
