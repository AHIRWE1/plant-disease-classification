import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app


class FakePredictor:
    class_names = ["Tomato___healthy", "Tomato___Early_blight"]

    def predict(self, image_bytes, top_k=3):
        return [
            {
                "class_name": "Tomato___healthy",
                "confidence": 0.9
            }
        ][:top_k]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client_context = TestClient(
            create_app(predictor_loader=lambda project_directory: FakePredictor())
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    @staticmethod
    def _png_bytes():
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color=(0, 120, 0)).save(
            buffer,
            format="PNG"
        )
        return buffer.getvalue()

    def test_health_reports_loaded_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["class_count"], 2)

    def test_predict_returns_top_class_and_confidence(self):
        response = self.client.post(
            "/predict?top_k=1",
            files={
                "image": (
                    "leaf.png",
                    self._png_bytes(),
                    "image/png"
                )
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["predictions"][0]["class_name"],
            "Tomato___healthy"
        )

    def test_predict_rejects_unsupported_media_type(self):
        response = self.client.post(
            "/predict",
            files={
                "image": (
                    "leaf.txt",
                    b"not an image",
                    "text/plain"
                )
            }
        )

        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
