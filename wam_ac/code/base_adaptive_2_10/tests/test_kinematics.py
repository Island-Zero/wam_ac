import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
from base_adaptive_2_10.kinematics import execution_position_mean_m


class KinematicsMetricTest(unittest.TestCase):
    def test_identity_is_zero_when_urdf_available(self):
        fastwam = Path(__import__("os").environ.get("FASTWAM_ROOT", ROOT.parents[1] / "repos/FastWAM"))
        urdf = fastwam / "third_party/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
        if not urdf.exists():
            self.skipTest("Set FASTWAM_ROOT to test with the RoboTwin URDF")
        qpos = np.zeros((32, 14), dtype=np.float32)
        self.assertEqual(execution_position_mean_m(qpos, qpos, urdf, 24), 0.0)

    def test_requires_14_controls(self):
        with self.assertRaises(ValueError):
            # Shape validation occurs before the missing path is parsed.
            execution_position_mean_m(np.zeros((32, 13)), np.zeros((32, 13)), "/missing.urdf", 24)


if __name__ == "__main__":
    unittest.main()
