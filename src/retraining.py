"""
Persist labelled uploads and safely evaluate retrained model candidates.
"""

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import uuid
import zipfile

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score

from src.model import compile_model, unfreeze_last_layers
from src.preprocessing import load_split_datasets


MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_FILES = 2000
MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_COMPRESSION_RATIO = 100
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _database_connection(project_directory):
    database_path = (
        Path(project_directory).resolve()
        / "data"
        / "retraining.sqlite3"
    )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_uploads (
            upload_id TEXT PRIMARY KEY,
            storage_path TEXT NOT NULL,
            status TEXT NOT NULL,
            image_count INTEGER NOT NULL,
            class_counts_json TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return connection


def _archive_members(archive_path, allowed_classes):
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("The uploaded file is not a valid ZIP archive.")

    with zipfile.ZipFile(archive_path) as archive:
        members = []
        total_uncompressed_size = 0
        file_infos = [
            info for info in archive.infolist()
            if not info.is_dir()
        ]
        if not file_infos:
            raise ValueError("The ZIP archive contains no files.")
        if len(file_infos) > MAX_ARCHIVE_FILES:
            raise ValueError(
                f"The ZIP archive exceeds the {MAX_ARCHIVE_FILES}-file limit."
            )

        for info in file_infos:
            archive_path_parts = PurePosixPath(info.filename)
            if (
                archive_path_parts.is_absolute()
                or "\\" in info.filename
                or any(part in ("", ".", "..") for part in archive_path_parts.parts)
            ):
                raise ValueError(
                    f"Unsafe path in ZIP archive: {info.filename}"
                )

            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError("Symbolic links are not allowed in uploaded ZIPs.")

            if archive_path_parts.parts[0] in allowed_classes:
                relative_parts = archive_path_parts.parts
            elif (
                len(archive_path_parts.parts) >= 2
                and archive_path_parts.parts[1] in allowed_classes
            ):
                relative_parts = archive_path_parts.parts[1:]
            else:
                raise ValueError(
                    f"Image path must start with a known class folder: "
                    f"{info.filename}"
                )

            if len(relative_parts) != 2:
                raise ValueError(
                    "Place each image directly inside its class folder."
                )
            class_name, filename = relative_parts
            if PurePosixPath(filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type in ZIP archive: {filename}"
                )
            if info.file_size > MAX_IMAGE_BYTES:
                raise ValueError(f"Image exceeds the 10 MB limit: {filename}")
            if info.file_size > 0 and info.compress_size == 0:
                raise ValueError(f"Invalid compressed image entry: {filename}")
            if (
                info.compress_size > 0
                and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
            ):
                raise ValueError(
                    f"Image compression ratio is too high: {filename}"
                )

            total_uncompressed_size += info.file_size
            if total_uncompressed_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "The ZIP archive exceeds the 500 MB uncompressed limit."
                )
            members.append((info, class_name))

        if not members:
            raise ValueError("The ZIP archive contains no supported images.")
        return members


def save_uploaded_dataset(project_directory, archive_path):
    """
    Validate labelled images, persist them by class, and record upload metadata.
    """
    project_directory = Path(project_directory).resolve()
    archive_path = Path(archive_path)
    class_metadata_path = (
        project_directory / "data" / "raw" / "dataset_metadata.json"
    )
    if not class_metadata_path.is_file():
        raise FileNotFoundError(
            f"Class metadata was not found: {class_metadata_path}"
        )
    if not archive_path.is_file():
        raise FileNotFoundError(f"Upload archive was not found: {archive_path}")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("The uploaded ZIP exceeds the 100 MB limit.")

    with class_metadata_path.open(encoding="utf-8") as file:
        class_names = json.load(file).get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Class metadata must contain a non-empty class_names list.")

    upload_id = uuid.uuid4().hex
    storage_root = project_directory / "data" / "uploaded" / upload_id
    allowed_classes = set(class_names)
    members = _archive_members(archive_path, allowed_classes)
    class_counts = {}

    storage_root.mkdir(parents=True)
    persisted = False
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info, class_name in members:
                image_bytes = archive.read(info)
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        if image.format not in ("JPEG", "PNG"):
                            raise ValueError(
                                f"File is not a JPEG or PNG image: {info.filename}"
                            )
                        if image.width * image.height > MAX_IMAGE_PIXELS:
                            raise ValueError(
                                f"Image dimensions exceed the supported limit: "
                                f"{info.filename}"
                            )
                        image.verify()
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        class_directory = storage_root / class_name
                        class_directory.mkdir(parents=True, exist_ok=True)
                        image.convert("RGB").save(
                            class_directory
                            / f"image_{class_counts.get(class_name, 0):04d}.jpg",
                            format="JPEG",
                            quality=95
                        )
                except (
                    UnidentifiedImageError,
                    Image.DecompressionBombError,
                    OSError
                ) as error:
                    raise ValueError(
                        f"Invalid image in ZIP archive: {info.filename}"
                    ) from error

                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if not class_counts:
            raise ValueError("The ZIP archive contains no usable labelled images.")

        now = datetime.now(timezone.utc).isoformat()
        connection = _database_connection(project_directory)
        try:
            connection.execute(
                """
                INSERT INTO dataset_uploads (
                    upload_id, storage_path, status, image_count,
                    class_counts_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    storage_root.relative_to(project_directory).as_posix(),
                    "uploaded",
                    sum(class_counts.values()),
                    json.dumps(class_counts, sort_keys=True),
                    now,
                    now
                )
            )
            connection.commit()
        finally:
            connection.close()
        persisted = True
    finally:
        if not persisted and storage_root.exists():
            shutil.rmtree(storage_root)

    return {
        "upload_id": upload_id,
        "status": "uploaded",
        "image_count": sum(class_counts.values()),
        "class_counts": class_counts
    }


def get_upload_record(project_directory, upload_id):
    """Return one upload record or raise if it does not exist."""
    connection = _database_connection(project_directory)
    try:
        record = connection.execute(
            "SELECT * FROM dataset_uploads WHERE upload_id = ?",
            (upload_id,)
        ).fetchone()
    finally:
        connection.close()
    if record is None:
        raise KeyError(f"Unknown upload ID: {upload_id}")

    result = dict(record)
    result["class_counts"] = json.loads(result.pop("class_counts_json"))
    if result["result_json"]:
        result["result"] = json.loads(result.pop("result_json"))
    else:
        result.pop("result_json")
    return result


def update_upload_record(project_directory, upload_id, **updates):
    if not updates:
        return
    allowed_columns = {
        "status",
        "result_json",
        "error"
    }
    if not set(updates).issubset(allowed_columns):
        raise ValueError("Unsupported upload record field.")

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    assignments = ", ".join(f"{column} = ?" for column in updates)
    values = list(updates.values()) + [upload_id]
    connection = _database_connection(project_directory)
    try:
        cursor = connection.execute(
            f"UPDATE dataset_uploads SET {assignments} WHERE upload_id = ?",
            values
        )
        connection.commit()
    finally:
        connection.close()
    if cursor.rowcount != 1:
        raise KeyError(f"Unknown upload ID: {upload_id}")


def trigger_retraining(project_directory, upload_id):
    """Mark a stored upload as queued, preventing duplicate concurrent jobs."""
    connection = _database_connection(project_directory)
    try:
        connection.execute("BEGIN IMMEDIATE")
        record = connection.execute(
            "SELECT status FROM dataset_uploads WHERE upload_id = ?",
            (upload_id,)
        ).fetchone()
        if record is None:
            raise KeyError(f"Unknown upload ID: {upload_id}")
        if record["status"] in ("queued", "training"):
            raise ValueError("Retraining is already queued for this upload.")

        active_job = connection.execute(
            """
            SELECT upload_id FROM dataset_uploads
            WHERE status IN ('queued', 'training')
            LIMIT 1
            """
        ).fetchone()
        if active_job is not None:
            raise ValueError("Another retraining job is already active.")

        connection.execute(
            """
            UPDATE dataset_uploads
            SET status = 'queued', error = NULL, updated_at = ?
            WHERE upload_id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), upload_id)
        )
        connection.commit()
    finally:
        connection.close()


def _evaluate_for_promotion(model, dataset):
    import tensorflow as tf

    labels = np.concatenate([
        batch_labels.numpy()
        for _, batch_labels in dataset
    ])
    probabilities = model.predict(dataset, verbose=0)
    metrics = model.evaluate(dataset, verbose=0, return_dict=True)
    return {
        "accuracy": float(accuracy_score(labels, np.argmax(probabilities, axis=1))),
        "loss": float(metrics["loss"])
    }


def should_promote_candidate(baseline_metrics, candidate_metrics):
    """Apply the pre-agreed validation-only model promotion rule."""
    required_metrics = {"accuracy", "loss"}
    if (
        not required_metrics.issubset(baseline_metrics)
        or not required_metrics.issubset(candidate_metrics)
    ):
        raise ValueError("Promotion metrics must include accuracy and loss.")

    return (
        candidate_metrics["accuracy"] >= baseline_metrics["accuracy"]
        and candidate_metrics["loss"] <= baseline_metrics["loss"] + 0.05
    )


def retrain_uploaded_dataset(project_directory, upload_id, epochs=2, batch_size=32):
    """
    Fine-tune the current model and promote only under the fixed rule:
    validation accuracy cannot fall and validation loss may rise at most 0.05.
    """
    import tensorflow as tf

    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    project_directory = Path(project_directory).resolve()
    record = get_upload_record(project_directory, upload_id)
    if record["status"] != "queued":
        raise ValueError("Retraining must be queued before it starts.")
    _update_upload_record(project_directory, upload_id, status="training")

    metadata_path = project_directory / "data" / "raw" / "dataset_metadata.json"
    with metadata_path.open(encoding="utf-8") as file:
        class_names = json.load(file)["class_names"]
    class_indices = {name: index for index, name in enumerate(class_names)}
    upload_path = project_directory / record["storage_path"]

    image_paths = []
    labels = []
    for class_name in sorted(record["class_counts"]):
        if class_name not in class_indices:
            raise ValueError(f"Unknown uploaded class: {class_name}")
        for image_path in sorted((upload_path / class_name).glob("*.jpg")):
            image_paths.append(str(image_path))
            labels.append(class_indices[class_name])
    if not image_paths:
        raise ValueError("No uploaded images are available for retraining.")

    datasets = load_split_datasets(
        project_directory,
        batch_size=batch_size
    )
    split_metadata_path = (
        project_directory / "data" / "processed" / "split_metadata.json"
    )
    with split_metadata_path.open(encoding="utf-8") as file:
        training_image_count = json.load(file)["images_per_split"]["train"]
    uploaded_dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def decode_and_preprocess(image_path, label):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, (224, 224))
        return preprocess_input(image), label

    uploaded_dataset = (
        uploaded_dataset
        .map(
            decode_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
    )
    training_dataset = (
        datasets["train"]
        .unbatch()
        .concatenate(uploaded_dataset.unbatch())
        .shuffle(
            buffer_size=record["image_count"] + training_image_count,
            seed=42,
            reshuffle_each_iteration=True
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model_path = project_directory / "models" / "plant_disease_classifier.keras"
    candidate_directory = project_directory / "models" / "candidates"
    candidate_directory.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_directory / f"{upload_id}.keras"

    candidate_model = tf.keras.models.load_model(model_path, compile=False)
    current_metrics = _evaluate_for_promotion(
        candidate_model,
        datasets["validation"]
    )
    base_model = next(
        layer
        for layer in candidate_model.layers
        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        )
    )
    unfreeze_last_layers(base_model, number_of_layers=20)
    compile_model(candidate_model, learning_rate=1e-5)
    candidate_model.fit(
        training_dataset,
        validation_data=datasets["validation"],
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1,
                restore_best_weights=True
            )
        ],
        verbose=2
    )

    candidate_metrics = _evaluate_for_promotion(
        candidate_model,
        datasets["validation"]
    )
    promoted = should_promote_candidate(
        current_metrics,
        candidate_metrics
    )
    result = {
        "baseline_validation": current_metrics,
        "candidate_validation": candidate_metrics,
        "promotion_rule": (
            "accuracy >= baseline and loss <= baseline + 0.05"
        ),
        "promoted": promoted
    }

    if promoted:
        versions_directory = project_directory / "models" / "versions"
        versions_directory.mkdir(parents=True, exist_ok=True)
        version_name = datetime.now(timezone.utc).strftime(
            "model_%Y%m%dT%H%M%SZ.keras"
        )
        shutil.copy2(
            model_path,
            versions_directory / version_name
        )
        candidate_model.save(candidate_path)
        os.replace(candidate_path, model_path)
        result["model_version"] = version_name
    else:
        candidate_model.save(candidate_path)
        result["candidate_path"] = candidate_path.relative_to(
            project_directory
        ).as_posix()

    _update_upload_record(
        project_directory,
        upload_id,
        status="promoted" if promoted else "rejected",
        result_json=json.dumps(result)
    )
    return result
