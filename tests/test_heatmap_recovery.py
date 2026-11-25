import tempfile
from pathlib import Path
import unittest

from heatmaps.generate_heatmap import (
    base_clip_result,
    build_checkpoint_signature,
    is_accelerator_failure,
    load_checkpoint,
    save_checkpoint,
)


class HeatmapRecoveryTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_input_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calibration = root / "calibration.json"
            rallies = root / "rallies.json"
            model = root / "model.pt"
            checkpoint = root / "partial.json"
            calibration.write_text("{}", encoding="utf-8")
            rallies.write_text("[]", encoding="utf-8")
            model.write_bytes(b"model")

            signature = build_checkpoint_signature(
                calibration,
                rallies,
                model,
                "gpu",
            )
            result = base_clip_result(
                root / "rally_157.mp4",
                {
                    "rally_157.mp4": {
                        "id": 157,
                        "start": 10.0,
                        "end": 20.0,
                        "set": 4,
                    }
                },
                "accelerator_error",
                "torch.AcceleratorError: CUDA error: unspecified launch failure",
            )
            save_checkpoint(checkpoint, signature, [result])
            self.assertEqual(load_checkpoint(checkpoint, signature), [result])

            model.write_bytes(b"changed")
            changed_signature = build_checkpoint_signature(
                calibration,
                rallies,
                model,
                "gpu",
            )
            self.assertEqual(load_checkpoint(checkpoint, changed_signature), [])

    def test_accelerator_failure_detection(self):
        self.assertTrue(
            is_accelerator_failure(
                RuntimeError(
                    "CUDA error: unspecified launch failure; hipErrorLaunchFailure"
                )
            )
        )
        self.assertTrue(
            is_accelerator_failure(
                RuntimeError("device-side assert triggered")
            )
        )
        self.assertFalse(is_accelerator_failure(ValueError("bad metadata")))


if __name__ == "__main__":
    unittest.main()
