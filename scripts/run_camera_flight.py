"""Bounded synthetic checkerboard flight with the frozen camera on the x500."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack, suppress
from pathlib import Path

from altiseek.config import load_config
from altiseek.deploy.assets import camera_sensor, checker_texture, export_tiles
from altiseek.imaging.config import CameraSpec
from altiseek.registry.run import sha256_file
from altiseek.world.spec import WorldSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-export", type=Path, help="Verified B export for the live detector pilot"
    )
    parser.add_argument("--search-pilot", action="store_true")
    parser.add_argument("--method", choices=("B5", "PPO"), default="B5")
    parser.add_argument("--training-seed", type=int, choices=(17, 29, 43), default=17)
    args = parser.parse_args()
    if args.search_pilot and args.world_export is None:
        parser.error("--search-pilot requires a B world export")
    repo = Path(__file__).resolve().parents[1]
    root = Path.home() / "altiseek-m10"
    source = root / "PX4-Autopilot/Tools/simulation/gz"
    build = root / "PX4-Autopilot/build/px4_sitl_default"
    with ExitStack() as stack:
        lock = stack.enter_context((root / "camera-flight.lock").open("w"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = root / "camera-flights" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run.mkdir(parents=True)
        (repo / "data/m10/camera-flight-path.txt").write_text(str(run) + "\n")
        camera = load_config(repo / "configs/camera/default.yaml", CameraSpec)
        spawn_x = spawn_y = 150.0
        if args.world_export is None:
            ground = export_tiles(checker_texture(), run / "textures", 0.25)
        else:
            spec = WorldSpec.load(args.world_export / "world.json")
            metadata = json.loads((args.world_export / "export.json").read_text())
            if (
                spec.split != "B"
                or metadata["world_id"] != spec.world_id
                or any(
                    sha256_file(args.world_export / name) != digest
                    for name, digest in metadata["sha256"].items()
                )
            ):
                raise ValueError("B world export identity/checksum failure")
            ground = (args.world_export / "ground.sdf").read_text()
            first = spec.targets[0]
            spawn_x = min(max(first.x_m, 25), spec.area_m - 25)
            spawn_y = min(max(first.y_m, 25), spec.area_m - 25)
        world = ET.parse(source / "worlds/default.sdf")
        element = world.getroot().find("world")
        element.set("name", "camera_flight")
        element.find("scene/ambient").text = "1 1 1 1"
        element.find("scene/shadows").text = "false"
        element.remove(element.find("light"))
        plane = element.find("model/link")
        plane.remove(plane.find("visual"))
        element.append(ET.fromstring(ground))
        plugins = ET.parse(source / "server.config").findall("plugins/plugin")
        for plugin in plugins:
            if not plugin.get("filename").startswith("gz-sim-"):
                continue
            plugin.attrib.pop("entity_name", None)
            plugin.attrib.pop("entity_type", None)
            if plugin.find("render_engine") is not None:
                plugin.find("render_engine").text = "ogre"
            element.append(plugin)
        model = ET.parse(source / "models/x500/model.sdf").getroot().find("model")
        model.set("name", "x500_0")
        ET.SubElement(model, "pose").text = f"{spawn_x} {spawn_y} 0.25 0 0 1.5707963267948966"
        link = ET.SubElement(model, "link", name="optical")
        ET.SubElement(
            link, "pose", relative_to="base_link"
        ).text = "0 0 -0.15 0 1.5707963267948966 0"
        link.append(ET.fromstring(camera_sensor(camera, "/altiseek/image", 2)))
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass").text = "0.001"
        inertia = ET.SubElement(inertial, "inertia")
        for field in ("ixx", "iyy", "izz"):
            ET.SubElement(inertia, field).text = "0.000001"
        joint = ET.SubElement(model, "joint", name="camera_joint", type="fixed")
        ET.SubElement(joint, "parent").text = "base_link"
        ET.SubElement(joint, "child").text = "optical"
        pose = ET.SubElement(
            model,
            "plugin",
            filename="gz-sim-pose-publisher-system",
            name="gz::sim::systems::PosePublisher",
        )
        for key, value in {
            "publish_link_pose": "false",
            "publish_model_pose": "true",
            "publish_nested_model_pose": "false",
            "use_pose_vector_msg": "true",
            "update_frequency": "50",
        }.items():
            ET.SubElement(pose, key).text = value
        element.append(model)
        sdf = run / "world.sdf"
        world.write(sdf, encoding="unicode")
        env = os.environ.copy()
        env.update(
            ROS_DOMAIN_ID="51",
            GZ_PARTITION="altiseek-camera-flight",
            GZ_IP="127.0.0.1",
            HEADLESS="1",
            PX4_SIM_MODEL="gz_x500",
            PX4_GZ_STANDALONE="1",
            PX4_GZ_WORLD="camera_flight",
            PX4_GZ_MODEL_NAME="x500_0",
            PX4_UXRCE_DDS_PORT="8899",
            PX4_PARAM_COM_RC_IN_MODE="4",
            PX4_PARAM_COM_RCL_EXCEPT="4",
            GZ_SIM_RESOURCE_PATH=str(source / "models"),
            LD_LIBRARY_PATH=str(root / "install/lib") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
        )
        jobs = []
        result = {"state": "starting", "run": str(run)}
        sources = [
            *sorted((repo / "src/altiseek").rglob("*.py")),
            *sorted((repo / "scripts").glob("*flight*.py")),
        ]
        provenance = {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
            ),
            "sources_sha256": {str(p.relative_to(repo)): sha256_file(p) for p in sources},
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "python": sys.version,
        }
        (run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

        def start(name: str, command: list[str], cwd: Path = repo) -> subprocess.Popen:
            log = stack.enter_context((run / f"{name}.log").open("w"))
            job = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            jobs.append(job)
            return job

        try:
            start("gazebo", ["gz", "sim", "-r", "-s", str(sdf)])
            start("agent", [str(root / "install/bin/MicroXRCEAgent"), "udp4", "-p", "8899"])
            start(
                "bridge",
                [
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "/altiseek/image@sensor_msgs/msg/Image[gz.msgs.Image",
                    "/model/x500_0/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
                ],
            )
            start("px4", [str(build / "bin/px4"), "-d", str(build / "etc")], run)
            start(
                "gcs",
                [
                    str(Path.home() / ".venvs/px4-build/bin/python"),
                    str(repo / "scripts/first_flight_gcs.py"),
                ],
            )
            if args.world_export is None:
                command = [sys.executable, str(repo / "scripts/camera_flight.py"), str(run)]
            else:
                command = [
                    sys.executable,
                    str(
                        repo
                        / (
                            "scripts/search_flight_pilot.py"
                            if args.search_pilot
                            else "scripts/perception_flight.py"
                        )
                    ),
                    str(run),
                    str(args.world_export / "world.json"),
                ]
            if args.search_pilot:
                command += ["--method", args.method, "--training-seed", str(args.training_seed)]
            controller = start("controller", command)
            result["state"] = "flight"
            (run / "status.json").write_text(json.dumps(result, indent=2))
            controller.wait(timeout=480 if args.search_pilot else 360)
            result.update(
                state="passed" if controller.returncode == 0 else "failed",
                controller_exit=controller.returncode,
            )
        except Exception as error:
            result.update(state="failed", error=repr(error))
            raise
        finally:
            for job in reversed(jobs):
                with suppress(ProcessLookupError):
                    os.killpg(job.pid, signal.SIGTERM)
            for job in jobs:
                try:
                    job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL)
                    job.wait(timeout=5)
            result["owned_processes_stopped"] = True
            (run / "status.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if result["state"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
