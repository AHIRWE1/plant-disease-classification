"""
Locust load profile for the plant-image prediction endpoint.
"""

import io
from PIL import Image
from locust import HttpUser, constant_pacing, task


def leaf_payload():
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color=(60, 125, 45)).save(
        buffer,
        format="JPEG"
    )
    return (
        "leaf.jpg",
        buffer.getvalue(),
        "image/jpeg"
    )


class PredictionUser(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def predict(self):
        filename, payload, content_type = leaf_payload()
        self.client.post(
            "/predict",
            files={
                "image": (
                    filename,
                    payload,
                    content_type
                )
            },
            timeout=30,
            name="/predict [single image]"
        )
