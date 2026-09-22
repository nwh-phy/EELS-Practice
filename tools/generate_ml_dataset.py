#!/usr/bin/env python3
"""Generate deterministic synthetic data for aberration-correction training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eels_sim import Config  # noqa: E402
from eels_sim.ml_dataset import generate_dataset, save_dataset  # noqa: E402


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--samples", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--max-order", type=int, choices=range(1, 6), default=5)
    result.add_argument("--max-active", type=int)
    result.add_argument("--amplitude-mev", type=float, default=90.0)
    result.add_argument("--input-mode", choices=("single", "probe-pair"), default="probe-pair")
    result.add_argument("--image-size", type=int, default=101)
    result.add_argument("--probe-term", default="D10")
    result.add_argument("--probe-mev", type=float, default=12.0)
    result.add_argument("--rays", type=int, default=8192)
    result.add_argument("--energy-half-range-mev", type=float, default=480.0)
    result.add_argument("--expected-counts", type=float, default=1_000_000.0)
    result.add_argument("--background-per-pixel", type=float, default=0.0)
    result.add_argument("--poisson", action="store_true")
    result.add_argument("--max-clipped-fraction", type=float, default=0.01)
    return result


def main():
    args = parser().parse_args()
    config = Config(
        n_rays=args.rays, energy_half_range_mev=args.energy_half_range_mev,
        energy_bins=401, y_bins=101, expected_counts=args.expected_counts,
        background_per_pixel=args.background_per_pixel, poisson=args.poisson)
    data = generate_dataset(
        args.samples, seed=args.seed, max_order=args.max_order,
        max_active=args.max_active, amplitude_mev=args.amplitude_mev,
        input_mode=args.input_mode, image_size=args.image_size,
        probe_term=args.probe_term, probe_mev=args.probe_mev, config=config,
        max_clipped_fraction=args.max_clipped_fraction)
    target = save_dataset(args.output, *data)
    print(f"wrote {args.samples} {args.input_mode} examples to {target}")


if __name__ == "__main__":
    main()
