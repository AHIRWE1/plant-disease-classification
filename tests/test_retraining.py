import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.retraining import (
    get_upload_record,
    save_uploaded_dataset,
    should_promote_candidate,
    trigger_retraining,
)


class RetrainingUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_directory = Path(self.temporary_directory.name)
        metadata_path = (
            self.project_directory
            / "data"
            / "raw"
            / "dataset_metadata.json"
        )
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_text(
            json.dumps({"class_names": ["crop___healthy"]}),
            encoding="utf-8"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _png_bytes():
        buffer = io.BytesIO()
        Image.fromarray(
            np.full((8, 8, 3), 80, dtype=np.uint8)
        ).save(buffer, format="PNG")
        return buffer.getvalue()

    def _write_archive(self, member_name, contents=None):
        archive_path = self.project_directory / "upload.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                member_name,
                self._png_bytes() if contents is None else contents
            )
        return archive_path

    def test_saves_labelled_images_and_records_database_metadata(self):
        archive_path = self._write_archive(
            "dataset/crop___healthy/leaf.png"
        )

        upload = save_uploaded_dataset(
            self.project_directory,
            archive_path
        )
        record = get_upload_record(
            self.project_directory,
            upload["upload_id"]
        )

        self.assertEqual(upload["status"], "uploaded")
        self.assertEqual(upload["image_count"], 1)
        self.assertEqual(
            record["class_counts"],
            {"crop___healthy": 1}
        )
        self.assertTrue(
            (
                self.project_directory
                / record["storage_path"]
                / "crop___healthy"
                / "image_0000.jpg"
            ).is_file()
        )

    def test_rejects_unknown_classes_and_unsafe_paths(self):
        for member_name in (
            "unknown_class/leaf.png",
            "../crop___healthy/leaf.png"
        ):
            archive_path = self._write_archive(member_name)
            with self.subTest(member_name=member_name):
                with self.assertRaises(ValueError):
                    save_uploaded_dataset(
                        self.project_directory,
                        archive_path
                    )

    def test_rejects_corrupt_images(self):
        archive_path = self._write_archive(
            "crop___healthy/leaf.png",
            contents=b"not an image"
        )

        with self.assertRaisesRegex(ValueError, "Invalid image"):
            save_uploaded_dataset(
                self.project_directory,
                archive_path
            )

    def test_queues_upload_once(self):
        upload = save_uploaded_dataset(
            self.project_directory,
            self._write_archive("crop___healthy/leaf.png")
        )

        trigger_retraining(
            self.project_directory,
            upload["upload_id"]
        )

        self.assertEqual(
            get_upload_record(
                self.project_directory,
                upload["upload_id"]
            )["status"],
            "queued"
        )
        with self.assertRaisesRegex(ValueError, "already queued"):
            trigger_retraining(
                self.project_directory,
                upload["upload_id"]
            )

    def test_candidate_promotion_uses_the_agreed_validation_threshold(self):
        baseline = {"accuracy": 0.90, "loss": 0.20}

        self.assertTrue(
            should_promote_candidate(
                baseline,
                {"accuracy": 0.90, "loss": 0.25}
            )
        )
        self.assertFalse(
            should_promote_candidate(
                baseline,
                {"accuracy": 0.899, "loss": 0.19}
            )
        )
        self.assertFalse(
            should_promote_candidate(
                baseline,
                {"accuracy": 0.91, "loss": 0.251}
            )
        )


if __name__ == "__main__":
    unittest.main()
