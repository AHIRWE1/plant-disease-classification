import io
import unittest

import numpy as np
from PIL import Image

from src.prediction import PlantDiseasePredictor


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


if __name__ == "__main__":
    unittest.main()
