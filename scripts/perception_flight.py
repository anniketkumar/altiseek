"""B-only live detector and shared belief smoke during a controlled hover.

Inference runs on a separate thread while ROS continues offboard heartbeats.
This is not a search episode or a transfer evaluation.
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rclpy
import torch
from camera_flight import CameraFlight
from PIL import Image
from sensor_msgs.msg import Image as RosImage

from altiseek.belief.state import Belief
from altiseek.cache.build import check_detector
from altiseek.cache.config import CacheConfig
from altiseek.calibration.model import SensorModel
from altiseek.config import load_config
from altiseek.deploy.perception import project_detections
from altiseek.deploy.projection import GroundCamera, quaternion_matrix
from altiseek.detector.infer import Detector
from altiseek.imaging.pipeline import render_view
from altiseek.world.sampler import prior_grid
from altiseek.world.spec import WorldSpec


class PerceptionFlight(CameraFlight):
    def __init__(self, output: Path, spec: WorldSpec) -> None:
        if spec.split != "B":
            raise ValueError("Perception integration uses B only")
        torch.set_num_threads(2)
        self.cfg = load_config("configs/cache/default.yaml", CacheConfig)
        self.detector = Detector(
            self.cfg.checkpoint,
            score_min=self.cfg.tau_min,
            iou=self.cfg.inference.iou,
            max_det=self.cfg.inference.max_det,
        )
        check_detector(self.detector, self.cfg)
        model = SensorModel.load(
            Path("data/sensor_models/fcf8cdae58ca3947.json"),
            detector_id=self.detector.detector_id,
            fit_world_set_id="5dc45d0e1ea5afbd",
        )
        prior = prior_grid(spec.prior_given, spec.area_m, spec.cell_m)
        self.belief = Belief(prior, len(spec.targets), len(self.cfg.render.levels.levels), model)
        self.initial_odds = self.belief.log_odds.copy()
        self.spec = spec
        self.inference_pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.records = []
        self.failures = []
        super().__init__(output)
        self.hold_s = 25

    def on_image(self, message: RosImage) -> None:
        before = len(self.images)
        super().on_image(message)
        if self.pending is not None and self.pending.done():
            try:
                self.records.append(self.pending.result())
            except Exception as error:
                self.failures.append(repr(error))
                self.start_landing("Live inference failed")
            self.pending = None
        if self.pending is None and len(self.images) > before and not self.failures:
            timestamp, rgb = self.images[-1]
            # Snapshot immutable pose records; do not share a mutating ROS list.
            poses = tuple(self.poses[-100:])
            self.pending = self.inference_pool.submit(self.process, timestamp, rgb, poses)

    def infer_frame(
        self, timestamp: float, rgb: np.ndarray, poses: tuple
    ) -> tuple[GroundCamera, np.ndarray, np.ndarray, dict]:
        started = time.monotonic()
        if not poses:
            raise ValueError("No matching camera pose")
        pose_stamp, xyz, quaternion = min(poses, key=lambda pose: abs(pose[0] - timestamp))
        delta = abs(pose_stamp - timestamp)
        if delta > 0.04:
            raise ValueError(f"Image/pose mismatch {delta:.3f} s")
        body = quaternion_matrix(np.array(quaternion))
        mount = quaternion_matrix(np.array([math.sqrt(0.5), 0, math.sqrt(0.5), 0]))
        origin = np.array(xyz) + body @ np.array([0, 0, -0.15])
        camera = GroundCamera(origin, body @ mount, 640, 80)
        view = render_view(rgb, 640, 640, self.cfg.render.camera.pipeline)
        detected = self.detector.detect(view[None])[0]
        evidence = project_detections(detected, camera, self.spec.cell_m, self.cfg.tau_min)
        index = f"{timestamp:.6f}"
        Image.fromarray(view).save(self.output / f"detector-{index}.png")
        metadata = {
            "stamp_s": timestamp,
            "pose_stamp_s": pose_stamp,
            "pose_delta_s": delta,
            "camera_enu_m": origin.tolist(),
            "detections": len(evidence),
            "swimmer_detections": int((evidence["class_id"] == 0).sum()),
            "elapsed_s": time.monotonic() - started,
        }
        return camera, evidence, view, metadata

    def process(self, timestamp: float, rgb: np.ndarray, poses: tuple) -> dict:
        started = time.monotonic()
        camera, evidence, _, metadata = self.infer_frame(timestamp, rgb, poses)
        grid = self.cfg.render.levels.grid_cells
        visible = camera.fully_visible_cells(grid, self.spec.cell_m)
        informative = self.belief.update(
            evidence, (0, 0, grid, grid), 0, 0, "V3", np.empty((0, 2)), 1, visible_cells=visible
        )
        np.savez_compressed(
            self.output / f"evidence-{timestamp:.6f}.npz",
            detections=evidence,
            visible=visible,
            probabilities=self.belief.probabilities,
        )
        return {
            **metadata,
            "informative_cells": informative,
            "visible_cells": int(visible.sum()),
            "elapsed_s": time.monotonic() - started,
        }

    def complete_perception(self) -> dict:
        if self.pending is not None:
            try:
                self.records.append(self.pending.result(timeout=30))
            except Exception as error:
                self.failures.append(repr(error))
        self.inference_pool.shutdown(wait=True)
        updated = int(np.count_nonzero(self.belief.log_odds != self.initial_odds))
        return {
            "passed": bool(
                self.passed and not self.failures and len(self.records) >= 10 and updated
            ),
            "world_id": self.spec.world_id,
            "split": "B",
            "scope": "hover integration only",
            "detector_id": self.detector.detector_id,
            "sensor_model_id": self.belief.model.sensor_model_id,
            "failures": self.failures,
            "frames_processed": len(self.records),
            "updated_cells": updated,
            "rows": self.records,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("world", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = PerceptionFlight(args.output, WorldSpec.load(args.world))
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
        report = node.complete_perception()
        (args.output / "perception-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in report.items() if key != "rows"}), flush=True
        )
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
