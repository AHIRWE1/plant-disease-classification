import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

from src.model import load_keras_model
from src.prediction import PlantDiseasePredictor, load_predictor


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray([probabilities])
        self.received_shape = None

    def predict(self, images, verbose=0):
        self.received_shape = images.shape
        return self.probabilities


class PredictionTests(unittest.TestCase):
    def _jpeg_bytes(self):
        image = Image.new("RGB", (20, 10), color=(30, 120, 45))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_preprocesses_image_and_returns_sorted_predictions(self):
        model = FakeModel([0.1, 0.7, 0.2])
        predictor = PlantDiseasePredictor(
            model=model,
            class_names=["class_a", "class_b", "class_c"]
        )

        mobilenet_module = types.ModuleType(
            "tensorflow.keras.applications.mobilenet_v2"
        )
        mobilenet_module.preprocess_input = lambda images: images
        applications_module = types.ModuleType(
            "tensorflow.keras.applications"
        )
        applications_module.mobilenet_v2 = mobilenet_module
        keras_module = types.ModuleType("tensorflow.keras")
        keras_module.applications = applications_module
        tensorflow_module = types.ModuleType("tensorflow")
        tensorflow_module.keras = keras_module
        tensorflow_modules = {
            "tensorflow": tensorflow_module,
            "tensorflow.keras": keras_module,
            "tensorflow.keras.applications": applications_module,
            "tensorflow.keras.applications.mobilenet_v2": mobilenet_module
        }
        with patch.dict(sys.modules, tensorflow_modules):
            predictions = predictor.predict(self._jpeg_bytes(), top_k=2)

        self.assertEqual(model.received_shape, (1, 224, 224, 3))
        self.assertEqual(
            [item["class_name"] for item in predictions],
            ["class_b", "class_c"]
        )
        self.assertAlmostEqual(predictions[0]["confidence"], 0.7)

    def test_rejects_non_image_and_oversized_input(self):
        predictor = PlantDiseasePredictor(
            model=FakeModel([1.0]),
            class_names=["class_a"]
        )

        with self.assertRaisesRegex(ValueError, "valid image"):
            predictor.predict(b"not an image")

        with self.assertRaisesRegex(ValueError, "10 MB"):
            predictor.predict(b"x" * (10 * 1024 * 1024 + 1))

    def test_loads_class_names_from_model_artifacts_without_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_directory = Path(temporary_directory)
            models_directory = project_directory / "models"
            models_directory.mkdir()
            (models_directory / "plant_disease_classifier.keras").touch()
            (models_directory / "class_names.json").write_text(
                json.dumps({"class_names": ["class_a", "class_b"]}),
                encoding="utf-8"
            )

            class LoadedModel:
                output_shape = (None, 2)

            tensorflow = types.ModuleType("tensorflow")
            tensorflow.keras = types.SimpleNamespace(
                models=types.SimpleNamespace(
                    load_model=lambda *args, **kwargs: LoadedModel()
                )
            )
            with patch.dict(sys.modules, {"tensorflow": tensorflow}):
                predictor = load_predictor(project_directory)

        self.assertEqual(predictor.class_names, ["class_a", "class_b"])

    def test_loads_windows_saved_keras_archive_with_portable_weight_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.keras"
            weights_buffer = io.BytesIO()
            with h5py.File(weights_buffer, "w") as weights:
                weights.create_dataset(
                    r"layers\functional\layers\dense\vars\0",
                    data=[1.0, 2.0]
                )
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("config.json", "{}")
                archive.writestr("metadata.json", "{}")
                archive.writestr(
                    "model.weights.h5",
                    weights_buffer.getvalue()
                )

            class LoadedModel:
                pass

            def load_model(path, compile):
                with zipfile.ZipFile(path) as archive:
                    with h5py.File(
                        io.BytesIO(archive.read("model.weights.h5")),
                        "r"
                    ) as weights:
                        self.assertIn("layers", weights)
                        self.assertIn("functional", weights["layers"])
                        self.assertIn(
                            "dense",
                            weights["layers/functional/layers"]
                        )
                return LoadedModel()

            tensorflow = types.ModuleType("tensorflow")
            tensorflow.keras = types.SimpleNamespace(
                models=types.SimpleNamespace(load_model=load_model)
            )
            with (
                patch("src.model.os", types.SimpleNamespace(name="posix")),
                patch.dict(sys.modules, {"tensorflow": tensorflow})
            ):
                model = load_keras_model(model_path, compile=False)

        self.assertIsInstance(model, LoadedModel)

    def test_uses_native_keras_loader_on_windows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.keras"
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("model.weights.h5", b"not inspected on Windows")

            expected_model = object()
            received = []
            tensorflow = types.ModuleType("tensorflow")
            tensorflow.keras = types.SimpleNamespace(
                models=types.SimpleNamespace(
                    load_model=lambda path, compile: (
                        received.append((path, compile)) or expected_model
                    )
                )
            )
            with (
                patch("src.model.os", types.SimpleNamespace(name="nt")),
                patch.dict(sys.modules, {"tensorflow": tensorflow})
            ):
                model = load_keras_model(model_path, compile=False)

        self.assertIs(model, expected_model)
        self.assertEqual(received, [(model_path, False)])


if __name__ == "__main__":
    unittest.main()
