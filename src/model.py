"""
MobileNetV2 transfer-learning model for PlantVillage classification.
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def build_model(
    number_of_classes,
    input_shape=(224, 224, 3),
    weights="imagenet",
    dropout_rate=0.2,
    learning_rate=1e-3
):
    """
    Create a frozen MobileNetV2 feature extractor and custom classifier head.
    """
    import tensorflow as tf

    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2.")
    if weights not in ("imagenet", None):
        raise ValueError("weights must be 'imagenet' or None.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in the range [0, 1).")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="leaf_image")
    features = base_model(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(features)
    features = tf.keras.layers.Dropout(
        dropout_rate,
        name="classifier_dropout"
    )(features)
    outputs = tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax",
        name="class_probabilities"
    )(features)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_classifier")
    compile_model(model, learning_rate)
    return model, base_model


def compile_model(model, learning_rate):
    """Compile the classifier for integer-encoded labels."""
    import tensorflow as tf

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def unfreeze_last_layers(base_model, number_of_layers=20):
    """
    Unfreeze the last requested non-BatchNormalization layers for fine-tuning.
    """
    import tensorflow as tf

    if number_of_layers < 1:
        raise ValueError("number_of_layers must be positive.")

    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False

    unfrozen_layers = 0
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            continue
        if unfrozen_layers == number_of_layers:
            break
        layer.trainable = True
        unfrozen_layers += 1

    if unfrozen_layers == 0:
        raise ValueError("The base model has no layers available to unfreeze.")

    return unfrozen_layers


def load_keras_model(model_path, compile=False):
    """Load a Keras model, normalizing Windows archive paths on Linux."""
    import h5py
    import tensorflow as tf

    model_path = Path(model_path)
    if os.name == "nt" or not zipfile.is_zipfile(model_path):
        return tf.keras.models.load_model(model_path, compile=compile)

    with tempfile.TemporaryDirectory() as temporary_directory:
        weights_path = Path(temporary_directory) / "model.weights.h5"
        with zipfile.ZipFile(model_path) as archive:
            if "model.weights.h5" not in archive.namelist():
                return tf.keras.models.load_model(model_path, compile=compile)
            with archive.open("model.weights.h5") as source:
                with weights_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

        with h5py.File(weights_path, "r") as source_weights:
            windows_path_found = []
            source_weights.visit(
                lambda name: windows_path_found.append(chr(92) in name)
            )
            uses_windows_paths = any(windows_path_found)
            if not uses_windows_paths:
                return tf.keras.models.load_model(model_path, compile=compile)

            normalized_weights_path = (
                Path(temporary_directory) / "normalized.weights.h5"
            )
            with h5py.File(normalized_weights_path, "w") as target_weights:
                for name, value in source_weights.attrs.items():
                    target_weights.attrs[name] = value
                _copy_hdf5_group(source_weights, target_weights)

        normalized_model_path = (
            Path(temporary_directory) / "normalized.keras"
        )
        with zipfile.ZipFile(model_path) as source_archive:
            with zipfile.ZipFile(normalized_model_path, "w") as target_archive:
                for file_info in source_archive.infolist():
                    with source_archive.open(file_info) as source:
                        with target_archive.open(file_info, "w") as target:
                            if file_info.filename == "model.weights.h5":
                                with normalized_weights_path.open("rb") as weights:
                                    shutil.copyfileobj(weights, target)
                            else:
                                shutil.copyfileobj(source, target)

        return tf.keras.models.load_model(
            normalized_model_path,
            compile=compile
        )


def _copy_hdf5_group(source_group, target_group):
    import h5py

    for name, item in source_group.items():
        path_parts = name.replace(chr(92), "/").split("/")
        target_parent = target_group
        for part in path_parts[:-1]:
            target_parent = target_parent.require_group(part)

        leaf_name = path_parts[-1]
        if isinstance(item, h5py.Group):
            target_item = target_parent.require_group(leaf_name)
            for attribute, value in item.attrs.items():
                target_item.attrs[attribute] = value
            _copy_hdf5_group(item, target_item)
        else:
            source_group.file.copy(
                item,
                target_parent,
                name=leaf_name
            )
