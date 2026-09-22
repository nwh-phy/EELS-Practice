#!/usr/bin/env python3
"""Predict offline effective-coefficient corrections from detector images."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eels_sim import TERMS  # noqa: E402
from eels_sim.ml_dataset import ODD_U_TERMS, display_image_to_float  # noqa: E402
from eels_sim.ml_model import build_alexnet_regressor, require_torch, select_device  # noqa: E402

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--probed-image", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch, _ = require_torch()
    device = select_device(torch, args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    metadata = checkpoint["metadata"]
    paths = [args.image]
    if metadata["input_mode"] == "probe-pair":
        if args.probed_image is None:
            raise SystemExit("this checkpoint requires --probed-image acquired after the documented probe")
        paths.append(args.probed_image)
    elif args.probed_image is not None:
        raise SystemExit("single-image checkpoint does not accept --probed-image")
    array = np.stack([
        display_image_to_float(path, metadata["image_size"]) for path in paths])
    model = build_alexnet_regressor(len(paths), len(TERMS)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(array[None]).to(device)).cpu().numpy()[0]
    correction = {name: float(value) for name, value in zip(TERMS, prediction)}
    result = {
        "effective_coefficient_correction_mev": correction,
        "scope": "offline model output; not a Nion actuator command",
        "input_mode": metadata["input_mode"],
        "probe": metadata.get("probe"),
        "model_metadata": {
            "dataset_version": metadata["dataset_version"],
            "forward_model_version": metadata["model_version"],
            "preprocessing_version": metadata["preprocessing_version"],
            "image_orientation": metadata["image_orientation"],
            "terms": metadata["terms"],
        },
    }
    if metadata["input_mode"] == "single":
        alternate = dict(correction)
        for name in ODD_U_TERMS:
            alternate[name] = -alternate[name]
        result["symmetry_equivalent_alternative_mev"] = alternate
        result["warning"] = "single-image odd-u signs are not identifiable in the symmetric simulator"
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
