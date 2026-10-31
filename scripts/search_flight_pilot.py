"""Three-view B5 stop-and-shoot B integration; not a transfer-scoring run."""

import argparse
import json
import queue
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from camera_flight import CameraFlight
from perception_flight import PerceptionFlight
from sb3_contrib import MaskablePPO
from sensor_msgs.msg import Image as RosImage
from tf2_msgs.msg import TFMessage

from altiseek.config import load_config
from altiseek.deploy.meter import BudgetExhausted, FlightMeter
from altiseek.deploy.search import LiveSearchPilot, LiveView
from altiseek.env.config import SearchConfig
from altiseek.geometry.frames import enu_to_ned
from altiseek.policies.baselines import Baseline
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec


class SearchFlight(PerceptionFlight):
    def __init__(
        self,
        output: Path,
        spec: WorldSpec,
        method: str,
        training_seed: int,
        measured: bool = False,
        normal_launch: bool = False,
        max_views: int = 3,
        battery_s: float = 500,
    ) -> None:
        super().__init__(output, spec)
        self.measured, self.normal_launch, self.max_views = measured, normal_launch, max_views
        self.search_config = load_config("configs/env/search.yaml", SearchConfig)
        self.search_config = self.search_config.model_copy(
            update={
                "start_level": 4 if normal_launch else 0,
                "energy": self.search_config.energy.model_copy(update={"battery_s": battery_s}),
            }
        )
        self.flight_meter = None
        self.settle_stamp = None
        self.settle_tag = None
        self.method, self.training_seed = method, training_seed
        self.checkpoint = None
        self.hold_s = 1000
        self.deadline_s = 1800 if measured else 420
        self.landing_timeout_s = 180
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.capture_requested = False
        self.stop_requested = False
        self.search_error = ""
        self.search_records = []
        self.policy_job = self.inference_pool.submit(self.run_search)

    def tick(self) -> None:
        if self.flight_meter is not None and self.flight_meter.snapshot()["exhausted"]:
            self.stop_requested = True
        if self.stop_requested and self.phase != "landing":
            self.start_landing(self.search_error)
        super().tick()

    def on_pose(self, message: TFMessage) -> None:
        super().on_pose(message)
        if not self.poses:
            return
        stamp_s, xyz, _ = self.poses[-1]
        if getattr(self, "settle_tag", None) != self.hold_since:
            self.settle_tag = self.hold_since
            self.settle_stamp = stamp_s if self.hold_since is not None else None
        meter = getattr(self, "flight_meter", None)
        if meter is not None:
            if not meter.snapshot()["started"] and self.hold_since is not None:
                meter.start(stamp_s, np.array(xyz))
            meter.update(stamp_s, np.array(xyz))

    def on_image(self, message: RosImage) -> None:
        if not self.capture_requested or self.hold_since is None:
            return
        if (
            self.settle_stamp is None
            or self.poses[-1][0] - self.settle_stamp < self.search_config.energy.dwell_s
        ):
            return
        previous = len(self.images)
        CameraFlight.on_image(self, message)
        if len(self.images) > previous:
            timestamp, rgb = self.images[-1]
            try:
                self.frames.put_nowait((timestamp, rgb, tuple(self.poses[-100:])))
                self.capture_requested = False
                # Images are already saved by the inference source; bound ROS RAM.
                self.images.clear()
            except queue.Full:
                pass

    def sense(self, goal: np.ndarray, level: int) -> LiveView:
        started = time.monotonic()
        while self.target is None or not self.poses or self.position is None:
            if time.monotonic() - started > 60 or self.done:
                raise TimeoutError("No flight telemetry for policy goal")
            time.sleep(0.02)
        if self.phase == "landing":
            raise RuntimeError("Flight is already landing")
        _, xyz, _ = self.poses[-1]
        p = self.position
        body_goal = goal + np.array([0, 0, 0.15])
        target = np.array([p.x, p.y, p.z]) + enu_to_ned(body_goal - xyz)
        self.target = target.tolist()
        self.altitude_m = float(body_goal[2])
        self.hold_since = None
        self.phase = "moving_to_policy_goal"
        self.capture_requested = True
        print(f"Policy goal L{level + 1}: {goal.tolist()}", flush=True)
        while True:
            if self.flight_meter is not None and self.flight_meter.snapshot()["exhausted"]:
                raise BudgetExhausted("Measured battery exhausted during waypoint execution")
            if self.phase == "landing" or self.done:
                raise RuntimeError("Flight ended before policy observation")
            if time.monotonic() - started > 120:
                raise TimeoutError("Policy waypoint capture timed out")
            try:
                timestamp, rgb, poses = self.frames.get(timeout=0.2)
                break
            except queue.Empty:
                continue
        camera, evidence, _, metadata = self.infer_frame(timestamp, rgb, poses)
        if self.flight_meter is not None and self.flight_meter.snapshot()["exhausted"]:
            raise BudgetExhausted("Measured battery exhausted during inference")
        error = float(np.linalg.norm(camera.position_enu_m - goal))
        if error > 1.0:
            raise ValueError(f"Actual camera missed policy goal by {error:.3f} m")
        metadata.update(
            goal_enu_m=goal.tolist(),
            level=level,
            goal_error_m=error,
            flight_and_inference_wall_s=time.monotonic() - started,
        )
        self.records.append(metadata)
        if self.flight_meter is not None:
            metadata["measured"] = self.flight_meter.snapshot()
        np.savez_compressed(self.output / f"detections-{timestamp:.6f}.npz", detections=evidence)
        return LiveView(camera, evidence)

    def run_search(self) -> None:
        try:
            config = self.search_config
            env = LiveSearchPilot(self.spec, self.cfg, self.belief.model, config, self.sense)
            if self.measured:
                self.flight_meter = FlightMeter(config.energy, env.nav.altitudes)
                env.meter = self.flight_meter
            options = {}
            if not self.normal_launch:
                target = self.spec.targets[0]
                options["start_node"] = env.nav.nearest(0, np.array([target.x_m, target.y_m]))
            policy = Baseline("B5", env.nav, env.sensor, "V3", seed=17, parameters={})
            model = None
            if self.method == "PPO":
                manifest = json.loads(Path("configs/eval/artifacts_v1.json").read_text())
                self.checkpoint = next(
                    p
                    for p in manifest["checkpoints"]
                    if p["variant"] == "V3" and p["seed"] == self.training_seed
                )
                if sha256_file(self.checkpoint["checkpoint"]) != self.checkpoint["sha256"]:
                    raise ValueError("Frozen PPO checkpoint checksum mismatch")
                model = MaskablePPO.load(self.checkpoint["checkpoint"], device="cpu")
                model.policy.set_training_mode(False)
                if (
                    model.observation_space != env.observation_space
                    or model.action_space != env.action_space
                    or model.policy.features_extractor.zero_belief
                ):
                    raise ValueError("Frozen PPO observation/action contract mismatch")
            rng = np.random.default_rng(1700 + self.training_seed)
            observation, info = env.reset(seed=17, options=options)
            self.search_records.append({"observation": 0, **info})
            for _ in range(self.max_views - 1 if self.max_views else config.max_steps):
                if info["terminated"]:
                    break
                mask = env.action_masks()
                if model is None:
                    action = policy.act(observation, mask)
                else:
                    tensors, _ = model.policy.obs_to_tensor(observation)
                    with torch.no_grad():
                        distribution = model.policy.get_distribution(tensors, action_masks=mask)
                        probabilities = (
                            distribution.distribution.probs.cpu().numpy()[0].astype(float)
                        )
                    action = int(
                        rng.choice(len(probabilities), p=probabilities / probabilities.sum())
                    )
                if not mask[action]:
                    raise ValueError("Policy selected a masked action")
                observation, _, terminated, truncated, info = env.step(action)
                self.search_records.append({"action": int(action), **info})
                np.savez_compressed(
                    self.output / f"belief-{env.observations}.npz",
                    probabilities=env.belief.probabilities,
                    map=observation["map"],
                    vector=observation["vector"],
                )
                if terminated or truncated:
                    break
        except Exception as error:
            self.search_error = repr(error)
        finally:
            if self.flight_meter is not None:
                self.flight_meter.stop()
            self.stop_requested = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("world", type=Path)
    parser.add_argument("--method", choices=("B5", "PPO"), default="B5")
    parser.add_argument("--training-seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--measured", action="store_true")
    parser.add_argument("--normal-launch", action="store_true")
    parser.add_argument("--max-views", type=int, default=3, help="0 runs until episode completion")
    parser.add_argument("--battery-s", type=float, default=500)
    args = parser.parse_args()
    rclpy.init()
    node = SearchFlight(
        args.output,
        WorldSpec.load(args.world),
        args.method,
        args.training_seed,
        args.measured,
        args.normal_launch,
        args.max_views,
        args.battery_s,
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.policy_job.result(timeout=10)
        node.inference_pool.shutdown(wait=True)
        completed = (args.max_views > 0 and len(node.records) >= args.max_views) or bool(
            node.search_records and node.search_records[-1]["ended"]
        )
        report = {
            "passed": bool(node.passed and completed and not node.search_error),
            "scope": "B integration; measured energy"
            if args.measured
            else "B integration; ideal energy only",
            "measured_energy": args.measured,
            "normal_edge_launch": args.normal_launch,
            "view_cap": args.max_views,
            "battery_s": args.battery_s,
            "meter": node.flight_meter.snapshot() if node.flight_meter is not None else None,
            "world_id": node.spec.world_id,
            "method": node.method,
            "training_seed": node.training_seed if node.method == "PPO" else None,
            "checkpoint": node.checkpoint,
            "error": node.search_error,
            "views": node.records,
            "search": node.search_records,
        }
        (args.output / "search-summary.json").write_text(
            json.dumps(
                report,
                indent=2,
                default=lambda value: (
                    value.tolist() if isinstance(value, np.ndarray) else value.item()
                ),
            )
            + "\n"
        )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("views", "search")},
                default=lambda value: (
                    value.tolist() if isinstance(value, np.ndarray) else value.item()
                ),
            ),
            flush=True,
        )
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
