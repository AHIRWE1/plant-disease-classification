
"""
data_acquisition.py

Downloads the PlantVillage dataset using TensorFlow Datasets,
exports the images into class-specific folders, and validates
the exported dataset.
"""

from pathlib import Path
import argparse
from collections import Counter
import json
import random

import tensorflow_datasets as tfds
from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------
# 1. CREATE PROJECT DIRECTORIES
# ---------------------------------------------------------

def create_directory(directory_path):
    """
    Create a directory if it does not already exist.

    Parameters:
        directory_path: Directory path to create.
    """
    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------
# 2. LOAD PLANTVILLAGE DATASET
# ---------------------------------------------------------

def load_plant_village_dataset(tfds_directory):
    """
    Download and load the complete PlantVillage dataset.

    TensorFlow Datasets manages the download and caching.
    """
    print("\nLoading PlantVillage dataset...")

    dataset, dataset_info = tfds.load(
        name="plant_village",
        split="train",
        data_dir=str(tfds_directory),
        with_info=True,
        as_supervised=False,
        shuffle_files=False
    )

    print("Dataset loaded successfully.")

    return dataset, dataset_info


# ---------------------------------------------------------
# 3. GET CLASS NAMES
# ---------------------------------------------------------

def get_class_names(dataset_info):
    """
    Extract the class names from dataset metadata.
    """
    class_names = dataset_info.features["label"].names

    print(f"\nNumber of classes: {len(class_names)}")

    return class_names


def _select_reservoir_indices(
    label_indices,
    number_of_classes,
    max_images_per_class,
    random_seed
):
    """
    Select a reproducible, unbiased sample of dataset indices per class.
    """
    if number_of_classes < 1:
        raise ValueError("number_of_classes must be positive.")
    if max_images_per_class < 1:
        raise ValueError("max_images_per_class must be positive.")

    rng = random.Random(random_seed)
    class_counts = Counter()
    selected_indices = {}

    for dataset_index, label_index in enumerate(label_indices):
        label_index = int(label_index)
        if not 0 <= label_index < number_of_classes:
            raise ValueError(f"Dataset label is out of range: {label_index}")

        class_counts[label_index] += 1
        reservoir = selected_indices.setdefault(label_index, [])

        if len(reservoir) < max_images_per_class:
            reservoir.append(dataset_index)
            continue

        replacement_index = rng.randrange(class_counts[label_index])
        if replacement_index < max_images_per_class:
            reservoir[replacement_index] = dataset_index

    return class_counts, selected_indices


# ---------------------------------------------------------
# 4. EXPORT DATASET INTO CLASS FOLDERS
# ---------------------------------------------------------

def export_dataset_to_folders(
    dataset,
    class_names,
    output_directory,
    max_images_per_class=200,
    random_seed=42
):
    """
    Export a seeded, class-balanced subset into class folders.

    Example:
        data/raw/
            Apple___Apple_scab/
            Apple___Black_rot/
            Apple___healthy/
    """

    output_directory = create_directory(output_directory)

    if any(output_directory.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_directory}. "
            "Move or remove its contents before acquiring a new dataset."
        )

    print("\nSelecting a reproducible subset per class...")
    available_counts, selected_indices = _select_reservoir_indices(
        (
            example["label"]
            for example in tfds.as_numpy(dataset)
        ),
        len(class_names),
        max_images_per_class,
        random_seed
    )

    selected_index_to_slot = {}
    for label_index, indices in selected_indices.items():
        for slot, dataset_index in enumerate(sorted(indices)):
            selected_index_to_slot[dataset_index] = (label_index, slot)

    if not selected_index_to_slot:
        raise ValueError("The dataset contains no examples to export.")

    class_counts = Counter()

    print("\nExporting selected images into class folders...")

    for index, example in enumerate(tfds.as_numpy(dataset)):
        selected = selected_index_to_slot.get(index)
        if selected is None:
            continue

        image_array = example["image"]
        label_index, slot = selected
        class_name = class_names[label_index]

        class_directory = create_directory(
            output_directory / class_name
        )

        image = Image.fromarray(image_array).convert("RGB")
        image_filename = f"image_{slot:04d}.jpg"

        image_path = class_directory / image_filename

        image.save(image_path, format="JPEG", quality=95)

        class_counts[class_name] += 1

    print("\nExport completed.")

    available_counts_by_name = {
        class_names[label_index]: count
        for label_index, count in available_counts.items()
    }

    return class_counts, available_counts_by_name


# ---------------------------------------------------------
# 5. SAVE DATASET METADATA
# ---------------------------------------------------------

def save_dataset_metadata(
    class_names,
    class_counts,
    output_directory,
    available_counts=None,
    max_images_per_class=200,
    random_seed=42
):
    """
    Save information about the exported dataset as JSON.
    """

    metadata = {
        "dataset_name": "PlantVillage",
        "source": "TensorFlow Datasets: plant_village",
        "number_of_classes": len(class_counts),
        "class_names": class_names,
        "total_images": sum(class_counts.values()),
        "images_per_class": dict(class_counts),
        "max_images_per_class": max_images_per_class,
        "random_seed": random_seed,
        "sampling_method": "seeded uniform sampling without replacement",
        "available_images_per_class": available_counts or {}
    }

    metadata_path = output_directory / "dataset_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nMetadata saved to: {metadata_path}")


# ---------------------------------------------------------
# 6. VALIDATE EXPORTED DATASET
# ---------------------------------------------------------

def validate_dataset(output_directory, class_names):
    """
    Check that all expected class folders exist and
    contain readable images.
    """

    output_directory = Path(output_directory)

    total_images = 0
    empty_classes = []
    invalid_images = []

    print("\nValidating exported dataset...")

    for class_name in class_names:

        class_directory = output_directory / class_name

        if not class_directory.exists():
            empty_classes.append(class_name)
            continue

        image_files = list(class_directory.glob("*.jpg"))

        if not image_files:
            empty_classes.append(class_name)

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    image.verify()

                total_images += 1

            except (
                OSError,
                SyntaxError,
                UnidentifiedImageError,
                ValueError
            ):
                invalid_images.append(str(image_path))

    print("\nDATASET VALIDATION REPORT")
    print("-------------------------")
    print(f"Expected classes: {len(class_names)}")
    print(f"Total valid images: {total_images}")
    print(f"Empty or missing classes: {len(empty_classes)}")
    print(f"Invalid images: {len(invalid_images)}")

    if empty_classes:
        print("\nEmpty or missing classes:")
        for class_name in empty_classes:
            print(class_name)

    if invalid_images:
        print("\nInvalid image files:")
        for image_path in invalid_images[:10]:
            print(image_path)

    return {
        "number_of_classes": len(class_names),
        "total_valid_images": total_images,
        "empty_classes": empty_classes,
        "invalid_images": invalid_images
    }


# ---------------------------------------------------------
# 7. MAIN DATA ACQUISITION PIPELINE
# ---------------------------------------------------------

def acquire_plant_village_dataset(
    project_directory,
    max_images_per_class=200,
    random_seed=42
):
    """
    Execute the complete dataset acquisition pipeline.
    """

    project_directory = Path(project_directory)

    tfds_directory = create_directory(
        project_directory / "data" / "tfds_cache"
    )

    raw_directory = create_directory(
        project_directory / "data" / "raw"
    )

    extracted_candidates = list(
        (tfds_directory / "downloads" / "extracted").glob(
            "*/Plant_leave_diseases_dataset_without_augmentation"
        )
    )

    try:
        dataset, dataset_info = load_plant_village_dataset(
            tfds_directory
        )
        class_names = get_class_names(dataset_info)
        class_counts, available_counts = export_dataset_to_folders(
            dataset,
            class_names,
            raw_directory,
            max_images_per_class=max_images_per_class,
            random_seed=random_seed
        )
    except (AssertionError, RuntimeError):
        if len(extracted_candidates) != 1:
            raise RuntimeError(
                "TFDS split generation failed and a unique extracted "
                "PlantVillage archive could not be found."
            )
        extracted_directory = extracted_candidates[0]

        print(
            "\nTFDS split generation did not complete. "
            "Using the extracted PlantVillage archive instead."
        )

        source_directories = sorted([
            directory
            for directory in extracted_directory.iterdir()
            if (
                directory.is_dir()
                and directory.name != "Background_without_leaves"
            )
        ])
        class_names = [
            directory.name
            for directory in source_directories
        ]
        if len(class_names) != 38:
            raise RuntimeError(
                "The extracted archive is not the documented 38-class "
                f"PlantVillage set (found {len(class_names)} classes)."
            )

        rng = random.Random(random_seed)
        class_counts = Counter()
        available_counts = {}
        raw_directory.mkdir(parents=True, exist_ok=True)

        for class_name, source_directory in zip(
            class_names,
            source_directories
        ):
            image_paths = sorted([
                path
                for path in source_directory.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower() in (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                )
            ])
            available_counts[class_name] = len(image_paths)

            selected_paths = image_paths
            if len(selected_paths) > max_images_per_class:
                selected_paths = rng.sample(
                    selected_paths,
                    max_images_per_class
                )

            class_directory = raw_directory / class_name
            class_directory.mkdir(parents=True)
            for slot, source_path in enumerate(
                sorted(selected_paths)
            ):
                with Image.open(source_path) as image:
                    image.convert("RGB").save(
                        class_directory / f"image_{slot:04d}.jpg",
                        format="JPEG",
                        quality=95
                    )
                class_counts[class_name] += 1

    # Save metadata
    save_dataset_metadata(
        class_names,
        class_counts,
        raw_directory,
        available_counts=available_counts,
        max_images_per_class=max_images_per_class,
        random_seed=random_seed
    )

    # Validate exported images
    validation_report = validate_dataset(
        raw_directory,
        class_names
    )

    if (
        validation_report["empty_classes"]
        or validation_report["invalid_images"]
    ):
        raise RuntimeError(
            "Dataset validation failed. Review the validation report "
            "before continuing."
        )

    return validation_report


# ---------------------------------------------------------
# 8. SCRIPT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire a capped PlantVillage image subset."
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=200,
        help="Maximum images to export for each class (default: 200)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for reproducible class sampling (default: 42)."
    )
    arguments = parser.parse_args()

    # Identify the main project directory
    PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]

    # Run acquisition
    report = acquire_plant_village_dataset(
        PROJECT_DIRECTORY,
        max_images_per_class=arguments.max_images_per_class,
        random_seed=arguments.seed
    )

    print("\nDataset acquisition process finished.")
    print(report)
