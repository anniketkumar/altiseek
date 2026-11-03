"""Measure Gazebo's 640 px nadir scale/orientation at all five fixed altitudes.

Requires the sourced ROS environment and pinned project Python. Synthetic
checkerboard only: this is a geometry/rendering gate, not held-out evaluation.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from contextlib import ExitStack, suppress
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage

from altiseek.config import load_config
from altiseek.deploy.assets import checker_texture, export_tiles, static_camera_world
from altiseek.geometry.levels import LevelTable, altitude_for_footprint_m
from altiseek.imaging.config import CameraSpec


def measure(rgb: np.ndarray, expected_gsd: float) -> dict:
    """5 m checker transitions and two asymmetric colour-marker quadrants."""
    gray = rgb.astype(float).mean(axis=2)
    r = len(rgb)
    spacing = []
    for line in (gray[r // 2 + 7], gray[:, r // 2 + 7]):
        edges = np.flatnonzero(np.diff((line > 127).astype(int)))
        if len(edges) < 4:
            raise ValueError("Checkerboard not visible")
        spacing.append(float(np.median(np.diff(edges))))
    red = (rgb[..., 0] > 180) & (rgb[..., 1] < 70) & (rgb[..., 2] < 70)
    blue = (rgb[..., 2] > 180) & (rgb[..., 0] < 70) & (rgb[..., 1] < 70)
    if not red.any() or not blue.any():
        raise ValueError("Orientation markers not visible")
    red_y, red_x = np.array(np.nonzero(red)).mean(axis=1)
    blue_y, blue_x = np.array(np.nonzero(blue)).mean(axis=1)
    gsd = [5 / s for s in spacing]
    errors = [abs(value / expected_gsd - 1) for value in gsd]
    oriented = red_y < r / 2 < blue_y and blue_x < r / 2 < red_x
    return {
        "gsd_xy_mpp": gsd,
        "expected_gsd_mpp": expected_gsd,
        "relative_error_xy": errors,
        "north_up": bool(oriented),
        "passed": bool(oriented and max(errors) < 0.03),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-engine", choices=("ogre2", "ogre"), default="ogre2")
    args = parser.parse_args()
    repo = Path.cwd()
    root = Path.home() / "altiseek-m10/camera-checks"
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((root / "probe.lock").open("w"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = root / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run.mkdir()
        pointer = repo / "data/m10/camera-check-path.txt"
        pointer.write_text(str(run) + "\n", encoding="utf-8")
        camera = load_config("configs/camera/default.yaml", CameraSpec)
        table = load_config("configs/levels/default.yaml", LevelTable)
        altitudes = [
            altitude_for_footprint_m(table.footprint_m(lvl), camera.hfov_deg)
            for lvl in table.levels
        ]
        ground = export_tiles(checker_texture(), run / "textures", 0.25)
        sdf = run / "world.sdf"
        sdf.write_text(
            static_camera_world(ground, camera, altitudes, args.render_engine), encoding="utf-8"
        )
        env = os.environ.copy()
        env.update(GZ_PARTITION="altiseek-camera-check", ROS_DOMAIN_ID="50", GZ_IP="127.0.0.1")
        os.environ["ROS_DOMAIN_ID"] = "50"
        jobs = []

        def start(name: str, command: list[str]) -> None:
            log = stack.enter_context((run / f"{name}.log").open("w"))
            jobs.append(
                subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            )

        rclpy.init()
        node = rclpy.create_node("altiseek_camera_check")
        frames: dict[int, list[np.ndarray]] = {i: [] for i in range(len(altitudes))}

        def receive(index: int, message: RosImage) -> None:
            if message.encoding not in ("rgb8", "bgr8"):
                raise ValueError(f"Unexpected camera encoding {message.encoding}")
            rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
            rgb = rows[:, : message.width * 3].reshape(message.height, message.width, 3).copy()
            if message.encoding == "bgr8":
                rgb = rgb[..., ::-1].copy()
            frames[index].append(rgb)
            frames[index] = frames[index][-3:]

        for i in frames:
            node.create_subscription(
                RosImage,
                f"/altiseek/probe/l{i}",
                lambda message, index=i: receive(index, message),
                qos_profile_sensor_data,
            )
        started = time.monotonic()
        report: dict = {"state": "running", "run": str(run), "render_engine": args.render_engine}
        try:
            start("gazebo", ["gz", "sim", "-r", "-s", str(sdf)])
            topics = [f"/altiseek/probe/l{i}@sensor_msgs/msg/Image[gz.msgs.Image" for i in frames]
            start("bridge", ["ros2", "run", "ros_gz_bridge", "parameter_bridge", *topics])
            while not all(len(images) >= 3 for images in frames.values()):
                if time.monotonic() - started > 120:
                    raise TimeoutError(
                        f"Camera frames received: {[len(v) for v in frames.values()]}"
                    )
                if any(job.poll() is not None for job in jobs):
                    raise RuntimeError("Gazebo or image bridge exited before capture")
                rclpy.spin_once(node, timeout_sec=0.2)
            rows = []
            for i, images in frames.items():
                Image.fromarray(images[-1]).save(run / f"L{i + 1}.png")
                row = measure(images[-1], table.footprint_m(table.levels[i]) / camera.resolution_px)
                rows.append({"level": i + 1, "altitude_m": altitudes[i], **row})
            report.update(
                state="passed" if all(row["passed"] for row in rows) else "failed", rows=rows
            )
        except Exception as error:
            report.update(state="failed", error=repr(error))
            raise
        finally:
            # Preserve received evidence even when another level times out.
            for i, images in frames.items():
                if images:
                    Image.fromarray(images[-1]).save(run / f"L{i + 1}.png")
            report["frame_counts"] = {str(i + 1): len(v) for i, v in frames.items()}
            for job in jobs:
                with suppress(ProcessLookupError):
                    os.killpg(job.pid, signal.SIGTERM)
            for job in jobs:
                try:
                    job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL)
                    job.wait(timeout=5)
            node.destroy_node()
            rclpy.shutdown()
            report["elapsed_s"] = time.monotonic() - started
            (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
        if report["state"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
