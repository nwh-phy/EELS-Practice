#!/usr/bin/env python3
"""Map a predicted coefficient correction to bounded control deltas offline."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eels_sim.control_mapping import (load_calibration,  # noqa: E402
                                      suggest_control_delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prediction = json.loads(args.prediction.read_text())
    correction = prediction.get("effective_coefficient_correction_mev")
    suggestion = suggest_control_delta(correction, load_calibration(args.calibration))
    text = json.dumps(suggestion, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
