"""
Train, fine-tune, and evaluate the PlantVillage classifier.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from src.model import build_model, compile_model, unfreeze_last_layers
from src.preprocessing import load_split_datasets


def evaluate_model(model, test_dataset, class_names, project_directory):
    """Evaluate once on the held-out test split and save its report."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    project_directory = Path(project_directory)
    models_directory = project_directory / "models"
    visualizations_directory = project_directory / "visualizations"
    models_directory.mkdir(parents=True, exist_ok=True)
    visualizations_directory.mkdir(parents=True, exist_ok=True)

    true_labels = np.concatenate([
        labels.numpy()
        for _, labels in test_dataset
    ])
    probabilities = model.predict(test_dataset, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        )
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0
        )
    )
    test_result = model.evaluate(test_dataset, verbose=0, return_dict=True)

    report = {
        "test_loss": float(test_result["loss"]),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names)))
        ).tolist(),
        "class_names": class_names
    }

    metrics_path = models_directory / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(report["confusion_matrix"]),
        display_labels=class_names
    )
    figure, axis = plt.subplots(figsize=(18, 18))
    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=90,
        colorbar=False,
        values_format="d"
    )
    figure.tight_layout()
    figure.savefig(
        visualizations_directory / "confusion_matrix.png",
        dpi=160
    )
    plt.close(figure)

    return report


def train_and_evaluate(
    project_directory,
    head_epochs=5,
    fine_tune_epochs=3,
    batch_size=32,
    fine_tune_layers=20,
    random_seed=42,
    weights="imagenet"
):
    """Train the classifier, optionally fine-tune it, and evaluate the test set."""
    import tensorflow as tf

    if head_epochs < 1:
        raise ValueError("head_epochs must be at least 1.")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs cannot be negative.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if fine_tune_epochs and fine_tune_layers < 1:
        raise ValueError("fine_tune_layers must be positive.")

    project_directory = Path(project_directory).resolve()
    tf.keras.utils.set_random_seed(random_seed)
    datasets = load_split_datasets(
        project_directory,
        batch_size=batch_size,
        random_seed=random_seed
    )

    metadata_path = project_directory / "data" / "raw" / "dataset_metadata.json"
    with metadata_path.open(encoding="utf-8") as file:
        class_names = json.load(file)["class_names"]

    model, base_model = build_model(
        number_of_classes=len(class_names),
        weights=weights
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    head_history = model.fit(
        datasets["train"],
        validation_data=datasets["validation"],
        epochs=head_epochs,
        callbacks=[early_stopping]
    )
    best_validation_loss = min(head_history.history["val_loss"])
    best_weights = model.get_weights()

    if fine_tune_epochs:
        unfrozen_count = unfreeze_last_layers(
            base_model,
            number_of_layers=fine_tune_layers
        )
        compile_model(model, learning_rate=1e-5)
        fine_tune_history = model.fit(
            datasets["train"],
            validation_data=datasets["validation"],
            epochs=fine_tune_epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True
                )
            ]
        )
        fine_tune_validation_loss = min(
            fine_tune_history.history["val_loss"]
        )
        if fine_tune_validation_loss < best_validation_loss:
            best_validation_loss = fine_tune_validation_loss
        else:
            model.set_weights(best_weights)
    else:
        unfrozen_count = 0

    models_directory = project_directory / "models"
    models_directory.mkdir(parents=True, exist_ok=True)
    model_path = models_directory / "plant_disease_classifier.keras"
    model.save(model_path)

    report = evaluate_model(
        model,
        datasets["test"],
        class_names,
        project_directory
    )
    report["best_validation_loss"] = float(best_validation_loss)
    report["fine_tuned_layers"] = unfrozen_count
    report["model_path"] = model_path.relative_to(
        project_directory
    ).as_posix()

    metrics_path = models_directory / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the PlantVillage image classifier."
    )
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-layers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet transfer learning, or random initialization for tests."
    )
    arguments = parser.parse_args()

    project_directory = Path(__file__).resolve().parents[1]
    training_report = train_and_evaluate(
        project_directory,
        head_epochs=arguments.head_epochs,
        fine_tune_epochs=arguments.fine_tune_epochs,
        batch_size=arguments.batch_size,
        fine_tune_layers=arguments.fine_tune_layers,
        random_seed=arguments.seed,
        weights=None if arguments.weights == "none" else arguments.weights
    )
    print(json.dumps(training_report, indent=2))
