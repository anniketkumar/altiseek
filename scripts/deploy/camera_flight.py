"""Capture timestamped camera/pose pairs during a bounded synthetic flight."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from check_deploy_camera import measure
from first_flight import FirstFlight
from geometry_msgs.msg import TransformStamped
from PIL import Image
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from tf2_msgs.msg import TFMessage

from altiseek.deploy.projection import GroundCamera, quaternion_matrix
from altiseek.geometry.camera import focal_length_px
from altiseek.geometry.levels import altitude_for_footprint_m


def stamp(message: RosImage | TransformStamped) -> float:
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


class CameraFlight(FirstFlight):
    def __init__(self, output: Path) -> None:
        # Camera is 0.15 m beneath the vehicle; actual pose still determines GSD.
        super().__init__(
            output, altitude_m=altitude_for_footprint_m(50, 80) + 0.15, hold_s=20, deadline_s=270
        )
        self.poses = []
        self.images = []
        self.heading_error_rad = math.inf
        self.tilt_rad = math.inf
        self.last_pose_wall = 0.0
        self.create_subscription(
            TFMessage, "/model/x500_0/pose", self.on_pose, qos_profile_sensor_data
        )
        self.create_subscription(
            RosImage, "/altiseek/image", self.on_image, qos_profile_sensor_data
        )

    def on_pose(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if transform.child_frame_id.split("::")[-1] != "x500_0":
                continue
            t, q = transform.transform.translation, transform.transform.rotation
            self.poses.append((stamp(transform), [t.x, t.y, t.z], [q.w, q.x, q.y, q.z]))
            rotation = quaternion_matrix(np.array([q.w, q.x, q.y, q.z]))
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            self.heading_error_rad = (yaw - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
            self.tilt_rad = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
            self.last_pose_wall = time.monotonic()
            if self.position is not None:
                # PX4's magnetic local heading can differ from Gazebo ENU north.
                # Correct the setpoint using measured world heading, not test outcomes.
                heading = self.position.heading + self.heading_error_rad
                self.yaw_ned_rad = (heading + math.pi) % (2 * math.pi) - math.pi

    def extra_settled(self) -> bool:
        return (
            time.monotonic() - self.last_pose_wall < 0.2
            and abs(self.heading_error_rad) < math.radians(1)
            and self.tilt_rad < math.radians(2)
        )

    def on_image(self, message: RosImage) -> None:
        if (
            self.phase != "holding"
            or self.hold_since is None
            or not self.extra_settled()
            or len(self.images) >= 50
        ):
            return
        if message.encoding not in ("rgb8", "bgr8"):
            self.start_landing(f"Unexpected image encoding {message.encoding}")
            return
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        rgb = rows[:, : message.width * 3].reshape(message.height, message.width, 3).copy()
        if message.encoding == "bgr8":
            rgb = rgb[..., ::-1].copy()
        self.images.append((stamp(message), rgb))

    def camera_report(self) -> dict:
        report = {"frames": len(self.images), "poses": len(self.poses), "passed": False}
        if not self.poses or len(self.images) < 10:
            return {**report, "error": "Insufficient camera or model-pose messages"}
        times = np.array([p[0] for p in self.poses])
        rows = []
        mount = quaternion_matrix(np.array([math.sqrt(0.5), 0, math.sqrt(0.5), 0]))
        for i, (timestamp, rgb) in enumerate(self.images):
            nearest = int(np.argmin(abs(times - timestamp)))
            pose_stamp, xyz, quaternion = self.poses[nearest]
            delta = abs(pose_stamp - timestamp)
            rotation = quaternion_matrix(np.array(quaternion))
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            heading_error = abs((yaw - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi)
            origin = np.array(xyz) + rotation @ np.array([0, 0, -0.15])
            camera = GroundCamera(origin, rotation @ mount, 640, 80)
            Image.fromarray(rgb).save(self.output / f"camera-{i:03d}.png")
            row = {
                "stamp_s": timestamp,
                "pose_stamp_s": pose_stamp,
                "pose_delta_s": delta,
                "camera_enu_m": origin.tolist(),
                "model_wxyz": quaternion,
                "heading_error_deg": math.degrees(heading_error),
            }
            try:
                geometry = measure(rgb, origin[2] / focal_length_px(80, 640))
                errors = []
                for channel, expected in ((0, [160, 160]), (2, [140, 140])):
                    other = [c for c in range(3) if c != channel]
                    mask = (rgb[..., channel] > 180) & (rgb[..., other] < 70).all(-1)
                    y, x = np.array(np.nonzero(mask)).mean(axis=1)
                    xy = camera.pixel_to_ground(np.array([x, y]))
                    errors.append(float(np.linalg.norm(xy - expected)))
                row.update(
                    geometry,
                    marker_error_m=errors,
                    passed=bool(
                        geometry["passed"]
                        and delta <= 0.04
                        and max(errors) <= 0.5
                        and heading_error < math.radians(1)
                    ),
                )
            except ValueError as error:
                row.update(passed=False, error=str(error))
            rows.append(row)
        report.update(rows=rows, passed=bool(self.passed and all(row["passed"] for row in rows)))
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = CameraFlight(args.output)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
        report = node.camera_report()
        (args.output / "camera-summary.json").write_text(json.dumps(report, indent=2) + "\n")
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
