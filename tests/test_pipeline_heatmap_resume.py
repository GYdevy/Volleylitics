import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_pipeline


class PipelineHeatmapResumeTests(unittest.TestCase):
    def test_resume_heatmap_skips_earlier_stages_and_restarts_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output_root = root / "output"
            match_dir = output_root / "match-test"
            clips_dir = match_dir / "rally_clips"
            clips_dir.mkdir(parents=True)

            video = root / "match.mp4"
            calibration = match_dir / "calibration.json"
            whistle_model = root / "whistle.pth"
            rally_model = root / "rally.pth"
            ball_model = root / "ball.pt"
            for path in (video, whistle_model, rally_model, ball_model):
                path.write_bytes(b"fixture")
            calibration.write_text(
                json.dumps(
                    {
                        "img_pts": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "net_left_x": 0,
                        "net_right_x": 1,
                        "net_y_img": 1,
                    }
                ),
                encoding="utf-8",
            )
            rallies = [{"start": 1.0, "end": 2.0, "set": 1}]
            (match_dir / "rallies.json").write_text(json.dumps(rallies), encoding="utf-8")
            (match_dir / "rallies_with_hitl.json").write_text(
                json.dumps(rallies), encoding="utf-8"
            )
            (match_dir / "rallies_with_clips.json").write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "start": 1.0,
                            "end": 2.0,
                            "set": 1,
                            "clip_path": "rally_clips/rally_000.mp4",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (clips_dir / "rally_000.mp4").write_bytes(b"clip")

            arguments = [
                "run_pipeline.py",
                "--video",
                str(video),
                "--match-id",
                "match-test",
                "--output-root",
                str(output_root),
                "--calibration",
                str(calibration),
                "--whistle-model",
                str(whistle_model),
                "--rally-model",
                str(rally_model),
                "--ball-model",
                str(ball_model),
                "--device",
                "gpu",
                "--resume-heatmap",
                "--heatmap-gpu-restarts",
                "2",
            ]
            launch_failure = subprocess.CalledProcessError(75, ["heatmap-worker"])
            with (
                patch.object(sys, "argv", arguments),
                patch.object(
                    run_pipeline,
                    "preflight",
                    return_value=("cuda", "test GPU"),
                ),
                patch.object(
                    run_pipeline,
                    "run_command",
                    side_effect=[launch_failure, None],
                ) as run_command,
            ):
                run_pipeline.EVENT_LOG_PATH = None
                self.assertEqual(run_pipeline.main(), 0)

            self.assertEqual(run_command.call_count, 2)
            for call in run_command.call_args_list:
                command = [str(value) for value in call.args[0]]
                self.assertIn("heatmaps.generate_heatmap", command)
                self.assertNotIn("rally_segmentator.split_rallies", command)

            state = json.loads(
                (match_dir / "pipeline_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["phase"], "complete")


if __name__ == "__main__":
    unittest.main()
