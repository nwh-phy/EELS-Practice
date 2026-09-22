"""Offline synthetic data for spectrometer-aberration regression.

This module deliberately contains no Nion hardware calls. It produces either
one image or a two-image observation in which the second image applies a known
coefficient probe. The probe breaks an exact left/right pupil symmetry that
makes some signed coefficients unidentifiable from one image in this model.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .model import MODEL_VERSION, POWERS, TERMS, Config, coefficients, simulate, terms_through


DATASET_VERSION = "eels-ml-synthetic-1"
PREPROCESSING_VERSION = "p99.8-sqrt-screen-y-v1"
INPUT_MODES = ("single", "probe-pair")
ODD_U_TERMS = tuple(name for name, (i, _) in zip(TERMS, POWERS) if i % 2)


def odd_u_reflection(values):
    """Return the coefficient vector related by the unobserved u -> -u symmetry."""
    result = coefficients(values)
    for name in ODD_U_TERMS:
        result[name] = -result[name]
    return result


def sample_coefficients(rng, max_order=5, max_active=None, amplitude_mev=90.0):
    """Draw a sparse-to-dense coefficient vector for synthetic training."""
    eligible = terms_through(max_order)
    if max_active is None:
        max_active = len(eligible)
    if type(max_active) is not int or not 1 <= max_active <= len(eligible):
        raise ValueError("max_active must be within the eligible coefficient count")
    if not np.isfinite(amplitude_mev) or not 0 < amplitude_mev <= 300:
        raise ValueError("amplitude_mev must be in (0, 300]")
    active_count = int(rng.integers(1, max_active + 1))
    selected = rng.choice(len(eligible), size=active_count, replace=False)
    result = coefficients()
    magnitudes = rng.uniform(0.1 * amplitude_mev, amplitude_mev, size=active_count)
    signs = rng.choice((-1.0, 1.0), size=active_count)
    for index, value in zip(selected, magnitudes * signs):
        result[eligible[int(index)]] = float(value)
    return result


def counts_to_uint8(counts, image_size=101):
    """Convert detector counts to the contrast-normalized network input."""
    if type(image_size) is not int or not 32 <= image_size <= 512:
        raise ValueError("image_size must be an integer in [32, 512]")
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("counts must be a finite non-negative 2D array")
    positive = values[values > 0]
    scale = float(np.percentile(positive, 99.8)) if positive.size else 1.0
    pixels = np.sqrt(np.clip(values / max(scale, np.finfo(float).tiny), 0, 1))
    image = Image.fromarray(np.rint(np.flipud(pixels) * 255).astype(np.uint8))
    return np.asarray(image.resize((image_size, image_size), Image.Resampling.BILINEAR))


def display_image_to_float(path, image_size=101):
    """Load an already contrast-mapped image in canonical screen orientation."""
    if type(image_size) is not int or not 32 <= image_size <= 512:
        raise ValueError("image_size must be an integer in [32, 512]")
    with Image.open(path) as source:
        image = source.convert("L").resize(
            (image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def render_observation(values, config, input_mode="probe-pair", image_size=101,
                       probe_term="D10", probe_mev=12.0):
    """Render one training observation and return it with clipping metadata."""
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    base_values = coefficients(values)
    base = simulate(base_values, config)
    channels = [counts_to_uint8(base.counts, image_size)]
    clipped = [base.clipped_fraction]
    if input_mode == "probe-pair":
        if probe_term not in TERMS or not np.isfinite(probe_mev) or probe_mev == 0:
            raise ValueError("probe-pair requires a known non-zero coefficient probe")
        probed_values = dict(base_values)
        probed_values[probe_term] += float(probe_mev)
        probed = simulate(probed_values, config)
        channels.append(counts_to_uint8(probed.counts, image_size))
        clipped.append(probed.clipped_fraction)
    return np.stack(channels), max(clipped)


def generate_dataset(sample_count, seed=1, max_order=5, max_active=None,
                     amplitude_mev=90.0, input_mode="probe-pair", image_size=101,
                     probe_term="D10", probe_mev=12.0, config=None,
                     max_clipped_fraction=0.01):
    """Generate arrays and metadata; rejected clipped frames are resampled."""
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    if not 0 <= max_clipped_fraction <= 1:
        raise ValueError("max_clipped_fraction must be in [0, 1]")
    config = Config(n_rays=8192, energy_half_range_mev=480, energy_bins=401,
                    y_bins=101) if config is None else config
    if not isinstance(config, Config):
        raise ValueError("config must be Config")
    rng = np.random.default_rng(seed)
    channels = 1 if input_mode == "single" else 2
    images = np.empty((sample_count, channels, image_size, image_size), dtype=np.uint8)
    corrections = np.empty((sample_count, len(TERMS)), dtype=np.float32)
    initial = np.empty_like(corrections)
    accepted = attempts = 0
    max_attempts = max(100, sample_count * 100)
    while accepted < sample_count and attempts < max_attempts:
        attempts += 1
        values = sample_coefficients(rng, max_order, max_active, amplitude_mev)
        sample_config = replace(config, noise_seed=(int(seed) + attempts) % (2**32))
        observation, clipped = render_observation(
            values, sample_config, input_mode, image_size, probe_term, probe_mev)
        if clipped > max_clipped_fraction:
            continue
        vector = np.asarray([values[name] for name in TERMS], dtype=np.float32)
        images[accepted] = observation
        initial[accepted] = vector
        corrections[accepted] = -vector
        accepted += 1
    if accepted != sample_count:
        raise RuntimeError(
            f"only generated {accepted}/{sample_count} examples after {attempts} attempts; "
            "increase the energy field or clipping threshold, or lower the amplitude")
    metadata = {
        "dataset_version": DATASET_VERSION,
        "model_version": MODEL_VERSION,
        "terms": list(TERMS),
        "powers": [list(power) for power in POWERS],
        "target": "effective coefficient correction in meV (-initial coefficients)",
        "input_mode": input_mode,
        "image_size": image_size,
        "preprocessing_version": PREPROCESSING_VERSION,
        "image_orientation": "screen rows top-to-bottom; energy columns left-to-right",
        "probe": {"term": probe_term, "value_mev": probe_mev} if input_mode == "probe-pair" else None,
        "odd_u_terms": list(ODD_U_TERMS),
        "single_image_identifiability": (
            "odd-u coefficients are identifiable only up to their simultaneous sign reflection"
            if input_mode == "single" else "known probe breaks the model's odd-u reflection symmetry"),
        "seed": int(seed),
        "sample_count": sample_count,
        "max_order": max_order,
        "max_active": max_active,
        "amplitude_mev": float(amplitude_mev),
        "max_clipped_fraction": float(max_clipped_fraction),
        "config": asdict(config),
        "scope": "offline synthetic effective coefficients; not Nion actuator settings",
    }
    return images, corrections, initial, metadata


def save_dataset(path, images, corrections, initial, metadata):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, images=np.asarray(images, dtype=np.uint8),
        corrections_mev=np.asarray(corrections, dtype=np.float32),
        initial_mev=np.asarray(initial, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    return target
