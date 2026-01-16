#!/usr/bin/env python3
"""
Script to create a sample_images directory with representative test images
from the MVTecAD dataset for use in the Gradio web UI.

For each category, it copies:
- 1 good (normal) image
- 2 images from each anomaly type (up to 4 anomaly types per category)
"""

import os
import shutil
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "data" / "MVTecAD"
SAMPLE_DIR = BASE_DIR / "sample_images"

# MVTec categories
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# Number of samples per type
NUM_GOOD_SAMPLES = 1
NUM_ANOMALY_SAMPLES_PER_TYPE = 2
MAX_ANOMALY_TYPES = 4  # Maximum number of different anomaly types to include


def get_anomaly_types(category_path: Path) -> list:
    """Get list of anomaly types for a category (excluding 'good')."""
    test_path = category_path / "test"
    if not test_path.exists():
        return []

    anomaly_types = [d.name for d in test_path.iterdir()
                     if d.is_dir() and d.name != "good"]
    return sorted(anomaly_types)


def copy_samples(src_dir: Path, dst_dir: Path, num_samples: int):
    """Copy a limited number of image samples from source to destination."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = [f for f in src_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    # Sort for consistency
    image_files.sort()

    # Copy only the requested number of samples
    for img_file in image_files[:num_samples]:
        shutil.copy2(img_file, dst_dir / img_file.name)


def create_sample_images():
    """Create sample images directory structure."""
    print(f"Creating sample images directory: {SAMPLE_DIR}")

    # Create main sample directory
    SAMPLE_DIR.mkdir(exist_ok=True)

    total_images = 0

    for category in CATEGORIES:
        category_src = DATASET_DIR / category
        category_dst = SAMPLE_DIR / category

        if not category_src.exists():
            print(f"⚠️  Skipping {category}: source directory not found")
            continue

        print(f"\n📁 Processing {category}...")

        # Copy good (normal) samples
        good_src = category_src / "test" / "good"
        good_dst = category_dst / "good"

        if good_src.exists():
            copy_samples(good_src, good_dst, NUM_GOOD_SAMPLES)
            num_good = len(list(good_dst.glob("*")))
            print(f"   ✓ Copied {num_good} good samples")
            total_images += num_good

        # Get all anomaly types for this category
        anomaly_types = get_anomaly_types(category_src)

        # Limit to MAX_ANOMALY_TYPES
        selected_anomalies = anomaly_types[:MAX_ANOMALY_TYPES]

        # Copy anomaly samples
        for anomaly_type in selected_anomalies:
            anomaly_src = category_src / "test" / anomaly_type
            anomaly_dst = category_dst / anomaly_type

            if anomaly_src.exists():
                copy_samples(anomaly_src, anomaly_dst, NUM_ANOMALY_SAMPLES_PER_TYPE)
                num_anomaly = len(list(anomaly_dst.glob("*")))
                print(f"   ✓ Copied {num_anomaly} {anomaly_type} samples")
                total_images += num_anomaly

        if len(anomaly_types) > MAX_ANOMALY_TYPES:
            skipped = len(anomaly_types) - MAX_ANOMALY_TYPES
            print(f"   ℹ️  Skipped {skipped} additional anomaly types: {', '.join(anomaly_types[MAX_ANOMALY_TYPES:])}")

    print(f"\n✅ Sample images directory created successfully!")
    print(f"📊 Total images copied: {total_images}")
    print(f"📂 Location: {SAMPLE_DIR}")


if __name__ == "__main__":
    create_sample_images()
