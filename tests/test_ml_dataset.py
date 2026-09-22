import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from eels_sim import Config, TERMS, simulate
from eels_sim.ml_dataset import (ODD_U_TERMS, counts_to_uint8,
                                 display_image_to_float, generate_dataset,
                                 odd_u_reflection, render_observation, save_dataset)


class MLDatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(n_rays=4096, energy_half_range_mev=480,
                             energy_bins=401, y_bins=81)

    def test_single_image_has_exact_odd_u_reflection_ambiguity(self):
        values = {name: (-1.0 if index % 2 else 1.0) * (index + 1)
                  for index, name in enumerate(TERMS)}
        reflected = odd_u_reflection(values)
        self.assertTrue(ODD_U_TERMS)
        np.testing.assert_allclose(
            simulate(values, self.config).expected,
            simulate(reflected, self.config).expected, atol=1e-12)

    def test_probe_pair_breaks_reflection_ambiguity(self):
        values = {"D10": 25, "D11": -14, "D02": 9, "D23": 6}
        direct, _ = render_observation(values, self.config, "probe-pair", 64, "D10", 12)
        reflected, _ = render_observation(
            odd_u_reflection(values), self.config, "probe-pair", 64, "D10", 12)
        np.testing.assert_array_equal(direct[0], reflected[0])
        self.assertFalse(np.array_equal(direct[1], reflected[1]))

    def test_count_preprocessing_shape_and_range(self):
        image = counts_to_uint8(simulate({"D01": 30}, self.config).counts, 73)
        self.assertEqual(image.shape, (73, 73))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), 0)

    def test_display_image_loader_preserves_trained_uint8_mapping(self):
        pixels = np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            Image.fromarray(pixels).save(path)
            loaded = display_image_to_float(path, 32)
        self.assertEqual(loaded.shape, (32, 32))
        self.assertEqual(float(loaded.min()), 0.0)
        self.assertEqual(float(loaded.max()), 1.0)

    def test_tiny_dataset_round_trip(self):
        arrays = generate_dataset(
            3, seed=12, max_order=2, max_active=2, amplitude_mev=20,
            input_mode="probe-pair", image_size=48, config=self.config,
            max_clipped_fraction=1)
        images, corrections, initial, metadata = arrays
        self.assertEqual(images.shape, (3, 2, 48, 48))
        self.assertEqual(corrections.shape, (3, len(TERMS)))
        np.testing.assert_allclose(corrections, -initial)
        self.assertEqual(metadata["scope"], "offline synthetic effective coefficients; not Nion actuator settings")
        with tempfile.TemporaryDirectory() as directory:
            target = save_dataset(Path(directory) / "tiny.npz", *arrays)
            archive = np.load(target, allow_pickle=False)
            self.assertEqual(json.loads(str(archive["metadata_json"]))["terms"], list(TERMS))


if __name__ == "__main__":
    unittest.main()
