import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import create_split_manifest, load_split_datasets


class SplitManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_directory = Path(self.temporary_directory.name)
        self.raw_directory = self.project_directory / "data" / "raw"
        self.raw_directory.mkdir(parents=True)

        self.class_names = ["crop___healthy", "crop___disease"]
        for class_name in self.class_names:
            class_directory = self.raw_directory / class_name
            class_directory.mkdir()
            for image_index in range(20):
                image = Image.fromarray(
                    np.full(
                        (8, 8, 3),
                        image_index * 10,
                        dtype=np.uint8
                    )
                )
                image.save(class_directory / f"image_{image_index:02d}.jpg")

        metadata = {
            "dataset_name": "PlantVillage",
            "class_names": self.class_names
        }
        with (self.raw_directory / "dataset_metadata.json").open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(metadata, file)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_creates_stratified_splits_without_copying_images(self):
        report = create_split_manifest(self.project_directory)
        manifest_path = (
            self.project_directory
            / "data"
            / "processed"
            / "split_manifest.csv"
        )

        self.assertEqual(report["total_images"], 40)
        self.assertEqual(
            report["images_per_split"],
            {"train": 28, "validation": 6, "test": 6}
        )
        for class_name in self.class_names:
            self.assertEqual(
                report["class_split_counts"][class_name],
                {"train": 14, "validation": 3, "test": 3}
            )
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(
            len(list(self.raw_directory.glob("*/*.jpg"))),
            40
        )

    def test_repeated_seed_produces_the_same_manifest(self):
        create_split_manifest(self.project_directory, random_seed=17)
        manifest_path = (
            self.project_directory
            / "data"
            / "processed"
            / "split_manifest.csv"
        )
        first_manifest = manifest_path.read_text(encoding="utf-8")

        create_split_manifest(self.project_directory, random_seed=17)
        second_manifest = manifest_path.read_text(encoding="utf-8")

        self.assertEqual(first_manifest, second_manifest)

    def test_loads_and_preprocesses_a_tensorflow_batch(self):
        create_split_manifest(self.project_directory)
        datasets = load_split_datasets(
            self.project_directory,
            image_size=(16, 16),
            batch_size=2
        )

        images, labels = next(iter(datasets["train"]))

        self.assertEqual(tuple(images.shape), (2, 16, 16, 3))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertGreaterEqual(float(images.numpy().min()), -1.0)
        self.assertLessEqual(float(images.numpy().max()), 1.0)

    def test_rejects_a_class_with_too_few_images(self):
        for image_path in sorted(
            (self.raw_directory / "crop___disease").glob("*.jpg")
        )[2:]:
            image_path.unlink()

        with self.assertRaises(ValueError):
            create_split_manifest(self.project_directory)


if __name__ == "__main__":
    unittest.main()
