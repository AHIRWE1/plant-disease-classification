import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.visualization import generate_dataset_visualizations


class DatasetVisualizationTests(unittest.TestCase):
    def test_creates_visuals_and_three_measured_interpretations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_directory = Path(temporary_directory)
            raw_directory = project_directory / "data" / "raw"
            metadata = raw_directory / "dataset_metadata.json"
            metadata.parent.mkdir(parents=True)

            class_names = ["crop___healthy", "crop___disease"]

            metadata.write_text(
                json.dumps({"class_names": class_names}),
                encoding="utf-8"
            )
            for class_index, class_name in enumerate(class_names):
                class_directory = raw_directory / class_name
                class_directory.mkdir()
                for image_index in range(2):
                    image = Image.fromarray(
                        np.full(
                            (8, 8, 3),
                            40 + class_index * 100 + image_index * 10,
                            dtype=np.uint8
                        )
                    )
                    image.save(
                        class_directory / f"image_{image_index}.jpg"
                    )

            report = generate_dataset_visualizations(project_directory)
            output_directory = project_directory / "visualizations"
            interpretation = (
                output_directory / "feature_interpretations.txt"
            ).read_text(encoding="utf-8")

            self.assertEqual(report["class_count"], 2)
            self.assertEqual(report["image_count"], 4)
            self.assertTrue(
                (output_directory / "class_distribution.png").is_file()
            )
            self.assertTrue(
                (output_directory / "class_samples.png").is_file()
            )
            self.assertTrue(
                (
                    output_directory
                    / "image_feature_comparisons.png"
                ).is_file()
            )
            self.assertIn("Mean brightness:", interpretation)
            self.assertIn("Green-channel ratio:", interpretation)
            self.assertIn("Edge contrast:", interpretation)


if __name__ == "__main__":
    unittest.main()
