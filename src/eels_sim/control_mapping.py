"""Offline mapping from effective aberrations to calibrated control deltas.

There are deliberately no Nion or hardware imports here. The caller must
provide a reviewed response matrix measured on the target instrument.
"""
import json
import math
from pathlib import Path

import numpy as np

from .model import TERMS


CALIBRATION_SCHEMA = "nion-response-matrix-1"
SUGGESTION_SCHEMA = "nion-control-suggestion-1"


def load_calibration(path):
    """Load a JSON calibration without accepting executable object formats."""
    return json.loads(Path(path).read_text())


def _finite_number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return float(value)


def _correction_vector(correction):
    if not isinstance(correction, dict) or set(correction) != set(TERMS):
        raise ValueError("correction must contain every canonical coefficient exactly once")
    return np.asarray([
        _finite_number(correction[name], f"correction.{name}") for name in TERMS])


def validate_calibration(calibration):
    """Validate and normalize the response-matrix calibration contract."""
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be a JSON object")
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(f"schema_version must be {CALIBRATION_SCHEMA}")
    if calibration.get("terms") != list(TERMS):
        raise ValueError("calibration terms must match the canonical order")
    actuators = calibration.get("actuators")
    if not isinstance(actuators, list) or not actuators:
        raise ValueError("calibration must define at least one actuator")
    normalized_actuators = []
    names = set()
    for index, actuator in enumerate(actuators):
        if not isinstance(actuator, dict):
            raise ValueError(f"actuators[{index}] must be an object")
        name, unit = actuator.get("name"), actuator.get("unit")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("actuator names must be non-empty and unique")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"actuator {name} must declare its unit")
        names.add(name)
        normalized_actuators.append({
            "name": name,
            "unit": unit,
            "max_abs_step": _finite_number(
                actuator.get("max_abs_step"), f"actuator {name} max_abs_step", positive=True),
        })
    response = np.asarray(calibration.get("response_mev_per_unit"), dtype=float)
    if response.shape != (len(TERMS), len(actuators)) or not np.isfinite(response).all():
        raise ValueError("response_mev_per_unit must be a finite terms-by-actuators matrix")
    if not np.any(response):
        raise ValueError("response matrix cannot be all zero")
    raw_weights = calibration.get("coefficient_weights", [1.0] * len(TERMS))
    weights = np.asarray(raw_weights, dtype=float)
    if weights.shape != (len(TERMS),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("coefficient_weights must contain one positive value per term")
    ridge = _finite_number(calibration.get("ridge", 0.01), "ridge", nonnegative=True)
    return normalized_actuators, response, weights, ridge


def suggest_control_delta(correction, calibration):
    """Return a bounded one-step control suggestion; never apply it to hardware."""
    target = _correction_vector(correction)
    actuators, response, weights, ridge = validate_calibration(calibration)
    step_limits = np.asarray([item["max_abs_step"] for item in actuators])

    # Solve in units of each actuator's allowed one-step change so that the
    # ridge penalty is dimensionless even when control units differ.
    normalized_response = response * step_limits[None, :]
    weighted_response = weights[:, None] * normalized_response
    weighted_target = weights * target
    augmented_matrix = np.vstack((weighted_response, ridge * np.eye(len(actuators))))
    augmented_target = np.concatenate((weighted_target, np.zeros(len(actuators))))
    normalized_delta = np.linalg.lstsq(
        augmented_matrix, augmented_target, rcond=None)[0]
    bounded_delta = np.clip(normalized_delta, -1.0, 1.0) * step_limits
    predicted = response @ bounded_delta
    residual = target - predicted
    saturated = [
        actuator["name"] for actuator, value in zip(actuators, normalized_delta)
        if abs(value) > 1.0
    ]
    return {
        "schema_version": SUGGESTION_SCHEMA,
        "control_delta": [
            {"name": actuator["name"], "unit": actuator["unit"], "value": float(value),
             "max_abs_step": actuator["max_abs_step"]}
            for actuator, value in zip(actuators, bounded_delta)
        ],
        "saturated_actuators": saturated,
        "predicted_effective_correction_mev": dict(zip(TERMS, map(float, predicted))),
        "unmodeled_residual_mev": dict(zip(TERMS, map(float, residual))),
        "weighted_residual_rms_mev": float(np.sqrt(np.mean((weights * residual) ** 2))),
        "ridge": ridge,
        "scope": "offline calibrated suggestion only; no instrument command was generated or applied",
    }
