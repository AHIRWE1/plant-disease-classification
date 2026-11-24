import unittest

from src.model import build_model, unfreeze_last_layers


class ModelTests(unittest.TestCase):
    def test_builds_classifier_with_expected_output_shape(self):
        model, base_model = build_model(
            number_of_classes=3,
            weights=None
        )

        self.assertEqual(model.output_shape, (None, 3))
        self.assertFalse(base_model.trainable)

    def test_fine_tuning_unfreezes_requested_layers_but_not_batch_norm(self):
        _, base_model = build_model(
            number_of_classes=3,
            weights=None
        )

        unfrozen_count = unfreeze_last_layers(
            base_model,
            number_of_layers=5
        )

        self.assertEqual(unfrozen_count, 5)
        self.assertEqual(
            sum(layer.trainable for layer in base_model.layers),
            5
        )


if __name__ == "__main__":
    unittest.main()
