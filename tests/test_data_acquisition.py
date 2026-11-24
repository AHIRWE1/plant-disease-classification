import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data_acquisition import (
    _select_reservoir_indices,
    export_dataset_to_folders,
)


class ReservoirSamplingTests(unittest.TestCase):
    def test_selects_no_more_than_the_cap_for_each_class(self):
        labels = [0] * 10 + [1] * 5 + [2] * 2

        counts, selected = _select_reservoir_indices(
            labels,
            number_of_classes=3,
            max_images_per_class=4,
            random_seed=42
        )

        self.assertEqual(counts, {0: 10, 1: 5, 2: 2})
        self.assertEqual(len(selected[0]), 4)
        self.assertEqual(len(selected[1]), 4)
        self.assertEqual(len(selected[2]), 2)
        self.assertTrue(all(index < 10 for index in selected[0]))
        self.assertTrue(all(10 <= index < 15 for index in selected[1]))
        self.assertTrue(all(15 <= index < 17 for index in selected[2]))

    def test_sampling_is_reproducible_for_a_fixed_seed(self):
        labels = [0] * 30 + [1] * 30

        first = _select_reservoir_indices(labels, 2, 5, 17)
        second = _select_reservoir_indices(labels, 2, 5, 17)

        self.assertEqual(first, second)

    def test_rejects_invalid_cap_and_label(self):
        with self.assertRaises(ValueError):
            _select_reservoir_indices([0], 1, 0, 42)

        with self.assertRaises(ValueError):
            _select_reservoir_indices([1], 1, 2, 42)


class DatasetExportTests(unittest.TestCase):
    def test_exports_only_the_selected_examples(self):
        examples = [
            {
                "image": np.full((4, 4, 3), index, dtype=np.uint8),
                "label": 0 if index < 5 else 1
            }
            for index in range(10)
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "raw"
            with patch(
                "src.data_acquisition.tfds.as_numpy",
                side_effect=lambda dataset: iter(dataset)
            ):
                class_counts, available_counts = export_dataset_to_folders(
                    examples,
                    ["class_a", "class_b"],
                    output_directory,
                    max_images_per_class=2,
                    random_seed=42
                )

            self.assertEqual(class_counts, {"class_a": 2, "class_b": 2})
            self.assertEqual(
                available_counts,
                {"class_a": 5, "class_b": 5}
            )
            exported_files = list(output_directory.glob("*/*.jpg"))
            self.assertEqual(len(exported_files), 4)

    def test_refuses_to_export_into_a_non_empty_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "raw"
            output_directory.mkdir()
            (output_directory / "existing.txt").write_text("keep")

            with self.assertRaises(FileExistsError):
                export_dataset_to_folders(
                    [],
                    ["class_a"],
                    output_directory
                )

            self.assertEqual(
                (output_directory / "existing.txt").read_text(),
                "keep"
            )


if __name__ == "__main__":
    unittest.main()
