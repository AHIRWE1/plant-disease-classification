"""
Generate PlantVillage class and image-feature visualizations.
"""

import argparse
import csv
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


FEATURE_NAMES = {
    "mean_brightness": "Mean brightness",
    "green_channel_ratio": "Green-channel ratio",
    "edge_contrast": "Edge contrast"
}


def _image_features(image_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize((64, 64)),
            dtype=np.float32
        ) / 255.0

    grayscale = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    green_ratio = rgb[:, :, 1] / (rgb.sum(axis=2) + 1e-7)
    vertical_edges = np.abs(np.diff(grayscale, axis=0)).mean()
    horizontal_edges = np.abs(np.diff(grayscale, axis=1)).mean()

    return {
        "mean_brightness": float(grayscale.mean()),
        "green_channel_ratio": float(green_ratio.mean()),
        "edge_contrast": float((vertical_edges + horizontal_edges) / 2)
    }


def _plot_class_distribution(class_counts, output_path):
    class_names = list(class_counts)
    counts = [class_counts[class_name] for class_name in class_names]

    figure, axis = plt.subplots(figsize=(16, 7))
    axis.bar(class_names, counts, color="forestgreen")
    axis.set_title("PlantVillage images per class")
    axis.set_xlabel("Plant disease class")
    axis.set_ylabel("Number of images")
    axis.tick_params(axis="x", labelrotation=90)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_class_samples(image_paths, class_names, output_path, random_seed):
    rng = random.Random(random_seed)
    columns = 5
    rows = (len(class_names) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 3 * rows),
        squeeze=False
    )

    for axis, class_name in zip(axes.flat, class_names):
        examples = image_paths[class_name]
        if not examples:
            axis.set_title(f"{class_name}\n(no images)")
            axis.axis("off")
            continue

        image_path = rng.choice(examples)
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(class_name, fontsize=8)
        axis.axis("off")

    for axis in axes.flat[len(class_names):]:
        axis.axis("off")

    figure.suptitle("One PlantVillage example per class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_feature_interpretations(class_features, output_path):
    healthy_classes = [
        row for row in class_features
        if row["condition"] == "healthy"
    ]
    diseased_classes = [
        row for row in class_features
        if row["condition"] == "diseased"
    ]
    if not healthy_classes or not diseased_classes:
        raise ValueError(
            "Feature interpretation requires both healthy and diseased classes."
        )

    lines = [
        "Image-feature interpretations",
        "=============================",
        "",
        "These descriptive comparisons use the mean for each class, so each "
        "class contributes equally.",
        "They describe this dataset and are not proof that a feature causes "
        "a disease or drives the neural network prediction.",
        ""
    ]

    for feature_name, label in FEATURE_NAMES.items():
        healthy_mean = float(np.mean([
            row[feature_name] for row in healthy_classes
        ]))
        diseased_mean = float(np.mean([
            row[feature_name] for row in diseased_classes
        ]))
        difference = diseased_mean - healthy_mean
        direction = (
            "higher in diseased classes"
            if difference > 1e-6
            else "higher in healthy classes"
            if difference < -1e-6
            else "similar between healthy and diseased classes"
        )

        lines.append(
            f"{label}: class-average healthy={healthy_mean:.4f}, "
            f"diseased={diseased_mean:.4f}; {direction} "
            f"(diseased minus healthy={difference:+.4f})."
        )

    Path(output_path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


def _plot_feature_comparisons(class_features, output_path):
    healthy_classes = [
        row for row in class_features
        if row["condition"] == "healthy"
    ]
    diseased_classes = [
        row for row in class_features
        if row["condition"] == "diseased"
    ]

    figure, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(15, 5))
    for axis, (feature_name, feature_label) in zip(
        axes,
        FEATURE_NAMES.items()
    ):
        axis.boxplot(
            [
                [row[feature_name] for row in healthy_classes],
                [row[feature_name] for row in diseased_classes]
            ]
        )
        axis.set_xticks(
            [1, 2],
            ["Healthy classes", "Diseased classes"]
        )
        axis.set_title(feature_label)
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Per-class image-feature means")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def generate_dataset_visualizations(project_directory, random_seed=42):
    """Create class-count, sample-image, and image-feature summaries."""
    project_directory = Path(project_directory).resolve()
    raw_directory = project_directory / "data" / "raw"
    metadata_path = raw_directory / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata was not found: {metadata_path}. "
            "Run data acquisition first."
        )

    with metadata_path.open(encoding="utf-8") as file:
        class_names = json.load(file).get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Dataset metadata must contain class names.")

    image_paths = {}
    class_counts = {}
    class_features = []
    for class_name in class_names:
        class_directory = raw_directory / class_name
        images = sorted(class_directory.glob("*.jpg"))
        if not images:
            raise ValueError(f"No JPEG images found for class '{class_name}'.")

        image_paths[class_name] = images
        class_counts[class_name] = len(images)
        image_feature_values = [
            _image_features(image_path)
            for image_path in images
        ]
        condition = (
            "healthy"
            if "healthy" in class_name.lower()
            else "diseased"
        )

        class_row = {
            "class_name": class_name,
            "condition": condition,
            "image_count": len(images)
        }
        for feature_name in FEATURE_NAMES:
            class_row[feature_name] = float(np.mean([
                values[feature_name]
                for values in image_feature_values
            ]))
        class_features.append(class_row)

    visualizations_directory = project_directory / "visualizations"
    visualizations_directory.mkdir(parents=True, exist_ok=True)

    _plot_class_distribution(
        class_counts,
        visualizations_directory / "class_distribution.png"
    )
    _plot_class_samples(
        image_paths,
        class_names,
        visualizations_directory / "class_samples.png",
        random_seed
    )

    summary_path = visualizations_directory / "image_feature_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "class_name",
                "condition",
                "image_count",
                *FEATURE_NAMES.keys()
            )
        )
        writer.writeheader()
        writer.writerows(class_features)

    _plot_feature_comparisons(
        class_features,
        visualizations_directory / "image_feature_comparisons.png"
    )
    _write_feature_interpretations(
        class_features,
        visualizations_directory / "feature_interpretations.txt"
    )

    return {
        "class_count": len(class_names),
        "image_count": sum(class_counts.values()),
        "visualizations_directory": visualizations_directory
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create PlantVillage class and feature visualizations."
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    project_directory = Path(__file__).resolve().parents[1]
    summary = generate_dataset_visualizations(
        project_directory,
        random_seed=arguments.seed
    )
    print(json.dumps(summary, indent=2, default=str))
