"""
Create reproducible train, validation, and test splits for PlantVillage.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


SPLIT_NAMES = ("train", "validation", "test")


def create_split_manifest(
    project_directory,
    train_fraction=0.70,
    validation_fraction=0.15,
    test_fraction=0.15,
    random_seed=42
):
    """
    Split images by class and write a portable CSV manifest.

    The image files stay in data/raw; the manifest records which split each
    image belongs to, avoiding an extra copy of the dataset.
    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("Each split fraction must be between 0 and 1.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1.")

    project_directory = Path(project_directory).resolve()
    raw_directory = project_directory / "data" / "raw"
    metadata_path = raw_directory / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata was not found: {metadata_path}. "
            "Run data acquisition first."
        )

    with metadata_path.open(encoding="utf-8") as file:
        dataset_metadata = json.load(file)

    class_names = dataset_metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Dataset metadata must contain a non-empty class_names list.")

    paths = []
    labels = []
    per_class_images = {}
    for label_index, class_name in enumerate(class_names):
        class_directory = raw_directory / class_name
        image_paths = sorted(class_directory.glob("*.jpg"))
        if len(image_paths) < 3:
            raise ValueError(
                f"Class '{class_name}' needs at least 3 images to appear "
                "in train, validation, and test splits."
            )
        per_class_images[class_name] = image_paths
        paths.extend(image_paths)
        labels.extend([label_index] * len(image_paths))

    if len(set(paths)) != len(paths):
        raise ValueError("An image path occurs more than once in the dataset.")

    validation_share_of_remainder = (
        validation_fraction / (train_fraction + validation_fraction)
    )

    train_validation_paths, test_paths, train_validation_labels, test_labels = (
        train_test_split(
            paths,
            labels,
            test_size=test_fraction,
            random_state=random_seed,
            stratify=labels
        )
    )

    train_paths, validation_paths, train_labels, validation_labels = (
        train_test_split(
            train_validation_paths,
            train_validation_labels,
            test_size=validation_share_of_remainder,
            random_state=random_seed,
            stratify=train_validation_labels
        )
    )

    split_rows = []
    for split_name, split_paths, split_labels in (
        ("train", train_paths, train_labels),
        ("validation", validation_paths, validation_labels),
        ("test", test_paths, test_labels)
    ):
        for image_path, label_index in zip(split_paths, split_labels):
            split_rows.append({
                "relative_path": image_path.relative_to(
                    project_directory
                ).as_posix(),
                "class_name": class_names[label_index],
                "label_index": label_index,
                "split": split_name
            })

    split_rows.sort(key=lambda row: row["relative_path"])
    split_counts = Counter(row["split"] for row in split_rows)
    class_split_counts = {
        class_name: {
            split_name: sum(
                row["class_name"] == class_name and row["split"] == split_name
                for row in split_rows
            )
            for split_name in SPLIT_NAMES
        }
        for class_name in class_names
    }

    processed_directory = project_directory / "data" / "processed"
    processed_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_directory / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("relative_path", "class_name", "label_index", "split")
        )
        writer.writeheader()
        writer.writerows(split_rows)

    split_metadata = {
        "dataset_name": dataset_metadata.get("dataset_name", "PlantVillage"),
        "random_seed": random_seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction
        },
        "total_images": len(split_rows),
        "images_per_split": dict(split_counts),
        "images_per_class": {
            class_name: len(images)
            for class_name, images in per_class_images.items()
        },
        "class_split_counts": class_split_counts,
        "manifest": manifest_path.relative_to(project_directory).as_posix()
    }
    split_metadata_path = processed_directory / "split_metadata.json"
    with split_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(split_metadata, file, indent=4)

    return split_metadata


def load_split_datasets(
    project_directory,
    manifest_path=None,
    image_size=(224, 224),
    batch_size=32,
    random_seed=42
):
    """
    Load manifest splits as MobileNetV2-preprocessed TensorFlow datasets.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    project_directory = Path(project_directory).resolve()
    if manifest_path is None:
        manifest_path = (
            project_directory / "data" / "processed" / "split_manifest.csv"
        )
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = project_directory / manifest_path

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest was not found: {manifest_path}. "
            "Create the manifest before loading datasets."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if (
        len(image_size) != 2
        or any(not isinstance(size, int) or size < 1 for size in image_size)
    ):
        raise ValueError("image_size must contain two positive integers.")

    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Split manifest is empty: {manifest_path}")

    datasets = {}
    for split_name in SPLIT_NAMES:
        split_rows = [row for row in rows if row["split"] == split_name]
        if not split_rows:
            raise ValueError(
                f"Split '{split_name}' has no images in {manifest_path}."
            )

        image_paths = [
            str(project_directory / Path(row["relative_path"]))
            for row in split_rows
        ]
        labels = [int(row["label_index"]) for row in split_rows]

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        if split_name == "train":
            dataset = dataset.shuffle(
                buffer_size=len(image_paths),
                seed=random_seed,
                reshuffle_each_iteration=True
            )

        def decode_and_preprocess(image_path, label):
            image_bytes = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image_bytes, channels=3)
            image = tf.image.resize(image, image_size)
            return preprocess_input(image), label

        datasets[split_name] = (
            dataset
            .map(
                decode_and_preprocess,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a reproducible train/validation/test manifest."
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    project_directory = Path(__file__).resolve().parents[1]
    split_report = create_split_manifest(
        project_directory,
        random_seed=arguments.seed
    )
    print(json.dumps(split_report, indent=2))
