import unittest

import numpy as np

from eels_sim import TERMS
from eels_sim.control_mapping import (CALIBRATION_SCHEMA,
                                      suggest_control_delta)


def calibration(response, step=10.0, ridge=0.0):
    actuator_count = np.asarray(response).shape[1]
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "terms": list(TERMS),
        "actuators": [
            {"name": f"reviewed-control-{index}", "unit": "test-unit",
             "max_abs_step": step}
            for index in range(actuator_count)
        ],
        "response_mev_per_unit": np.asarray(response, dtype=float).tolist(),
        "ridge": ridge,
    }


class ControlMappingTests(unittest.TestCase):
    def test_identity_calibration_recovers_requested_delta(self):
        correction = dict.fromkeys(TERMS, 0.0)
        correction["D10"], correction["D01"] = 2.5, -3.0
        result = suggest_control_delta(correction, calibration(np.eye(len(TERMS))))
        values = [item["value"] for item in result["control_delta"]]
        np.testing.assert_allclose(values[:2], [2.5, -3.0], atol=1e-12)
        self.assertAlmostEqual(result["weighted_residual_rms_mev"], 0.0)
        self.assertEqual(result["saturated_actuators"], [])

    def test_step_limit_is_applied_and_reported(self):
        response = np.zeros((len(TERMS), 1))
        response[0, 0] = 2.0
        correction = dict.fromkeys(TERMS, 0.0)
        correction[TERMS[0]] = 10.0
        result = suggest_control_delta(correction, calibration(response, step=1.0))
        self.assertEqual(result["control_delta"][0]["value"], 1.0)
        self.assertEqual(result["saturated_actuators"], ["reviewed-control-0"])
        self.assertEqual(result["unmodeled_residual_mev"][TERMS[0]], 8.0)

    def test_rejects_unknown_term_order_and_missing_limits(self):
        response = np.eye(len(TERMS))
        bad_terms = calibration(response)
        bad_terms["terms"] = list(reversed(TERMS))
        with self.assertRaises(ValueError):
            suggest_control_delta(dict.fromkeys(TERMS, 0.0), bad_terms)
        bad_limit = calibration(response)
        del bad_limit["actuators"][0]["max_abs_step"]
        with self.assertRaises(ValueError):
            suggest_control_delta(dict.fromkeys(TERMS, 0.0), bad_limit)


if __name__ == "__main__":
    unittest.main()
