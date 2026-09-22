#!/usr/bin/env python3
"""Train an AlexNet-style coefficient-correction regressor with PyTorch."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eels_sim import TERMS  # noqa: E402
from eels_sim.ml_dataset import ODD_U_TERMS  # noqa: E402
from eels_sim.ml_model import build_alexnet_regressor, require_torch, select_device  # noqa: E402


def load_dataset(path, torch):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        image_values = archive["images"].astype(np.float32) / 255.0
        target_values = archive["corrections_mev"].astype(np.float32)
    if tuple(metadata["terms"]) != TERMS:
        raise ValueError(f"{path}: coefficient order does not match this simulator")
    expected_channels = 1 if metadata["input_mode"] == "single" else 2
    expected_shape = (
        len(image_values), expected_channels, metadata["image_size"], metadata["image_size"])
    if image_values.shape != expected_shape:
        raise ValueError(f"{path}: image array does not match its metadata")
    if target_values.shape != (len(image_values), len(TERMS)):
        raise ValueError(f"{path}: correction target shape is invalid")
    images = torch.from_numpy(image_values)
    targets = torch.from_numpy(target_values)
    return torch.utils.data.TensorDataset(images, targets), metadata


def per_sample_loss(torch, prediction, target, scale, symmetry_aware):
    direct = torch.nn.functional.smooth_l1_loss(
        prediction / scale, target / scale, reduction="none").mean(dim=1)
    if not symmetry_aware:
        return direct
    reflected = target.clone()
    indices = [TERMS.index(name) for name in ODD_U_TERMS]
    reflected[:, indices] *= -1
    alternate = torch.nn.functional.smooth_l1_loss(
        prediction / scale, reflected / scale, reduction="none").mean(dim=1)
    return torch.minimum(direct, alternate)


def evaluate(torch, model, loader, device, scale, symmetry_aware):
    model.eval()
    total_loss = count = 0
    absolute = torch.zeros(len(TERMS), device=device)
    odd_indices = [TERMS.index(name) for name in ODD_U_TERMS]
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            prediction = model(images)
            losses = per_sample_loss(torch, prediction, targets, scale, symmetry_aware)
            total_loss += float(losses.sum().cpu())
            count += len(images)
            if symmetry_aware:
                reflected = targets.clone()
                reflected[:, odd_indices] *= -1
                choose_reflected = ((prediction - reflected).abs().mean(1)
                                    < (prediction - targets).abs().mean(1))
                targets = torch.where(choose_reflected[:, None], reflected, targets)
            absolute += (prediction - targets).abs().sum(0)
    return total_loss / count, (absolute / count).cpu().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch, _ = require_torch()
    torch.manual_seed(args.seed)
    train_set, train_meta = load_dataset(args.train, torch)
    val_set, val_meta = load_dataset(args.validation, torch)
    for key in ("input_mode", "image_size", "preprocessing_version", "terms", "probe"):
        if train_meta[key] != val_meta[key]:
            raise ValueError(f"training and validation metadata differ for {key}")
    device = select_device(torch, args.device)
    model = build_alexnet_regressor(train_set.tensors[0].shape[1], len(TERMS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)
    scale = float(train_meta["amplitude_mev"])
    symmetry_aware = train_meta["input_mode"] == "single"
    history = []
    best = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = seen = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            loss = per_sample_loss(torch, prediction, targets, scale, symmetry_aware).mean()
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(images)
            seen += len(images)
        val_loss, mae = evaluate(torch, model, val_loader, device, scale, symmetry_aware)
        row = {"epoch": epoch, "train_loss": running / seen, "validation_loss": val_loss,
               "validation_mae_mev": dict(zip(TERMS, mae))}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_loss < best:
            best = val_loss
            torch.save({
                "state_dict": model.state_dict(), "metadata": train_meta,
                "training": {"epoch": epoch, "best_validation_loss": best,
                             "symmetry_aware_loss": symmetry_aware,
                             "architecture": "alexnet-regressor-101-v1"},
            }, args.output)
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps({"best_validation_loss": best, "history": history}, ensure_ascii=False, indent=2) + "\n")
    print(f"best checkpoint: {args.output} on {device}")


if __name__ == "__main__":
    main()
