"""
Image validation, preprocessing, and model inference utilities.
"""

from dataclasses import dataclass
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}
IMAGE_SIZE = (224, 224)


@dataclass
class PlantDiseasePredictor:
    model: object
    class_names: list

    def predict(self, image_bytes, top_k=3):
        """Return the highest-probability classes for one encoded image."""
        if not image_bytes:
            raise ValueError("The uploaded image is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the 10 MB size limit.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        top_k = min(top_k, len(self.class_names))

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.format not in ALLOWED_IMAGE_FORMATS:
                    raise ValueError("Upload a JPEG or PNG image.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions exceed the supported limit.")
                image = image.convert("RGB").resize(IMAGE_SIZE)
                image_array = np.asarray(image, dtype=np.float32)
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError
        ) as error:
            raise ValueError("The uploaded file is not a valid image.") from error

        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        batch = preprocess_input(np.expand_dims(image_array, axis=0))
        probabilities = np.asarray(self.model.predict(batch, verbose=0))[0]
        if probabilities.shape != (len(self.class_names),):
            raise RuntimeError(
                "The model output does not match the saved class metadata."
            )

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        return [
            {
                "class_name": self.class_names[int(index)],
                "confidence": float(probabilities[index])
            }
            for index in top_indices
        ]


def load_predictor(project_directory):
    """Load the saved Keras model and class names from project artifacts."""
    import tensorflow as tf

    project_directory = Path(project_directory).resolve()
    model_path = project_directory / "models" / "plant_disease_classifier.keras"
    metadata_path = project_directory / "data" / "raw" / "dataset_metadata.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Class metadata not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        class_names = json.load(file).get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Class metadata must contain a non-empty class_names list.")

    model = tf.keras.models.load_model(model_path, compile=False)
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The saved model output count does not match class metadata."
        )
    return PlantDiseasePredictor(model=model, class_names=class_names)
