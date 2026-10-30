"""Bounded ROS 2/PX4 SITL check: take off, hold at 5 m, land and disarm.

Run only through run_first_flight.py, which owns an isolated local simulator.
Uses the system ROS Python environment, not the training environment.
"""

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLandDetected,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from altiseek.geometry.frames import enu_to_ned, enu_yaw_to_ned


def topic(direction: str, name: str, message: type) -> str:
    version = getattr(message, "MESSAGE_VERSION", 0)
    return f"/fmu/{direction}/{name}" + (f"_v{version}" if version else "")


class FirstFlight(Node):
    def __init__(self, output: Path) -> None:
        super().__init__("altiseek_first_flight")
        self.output = output
        self.stream = (output / "telemetry.jsonl").open("w", encoding="utf-8")
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.position: VehicleLocalPosition | None = None
        self.status: VehicleStatus | None = None
        self.landed: VehicleLandDetected | None = None
        self.last_position = 0.0
        self.last_status = 0.0
        self.last_land = 0.0
        for name, cls, callback in (
            ("vehicle_local_position", VehicleLocalPosition, self.on_position),
            ("vehicle_status", VehicleStatus, self.on_status),
            ("vehicle_land_detected", VehicleLandDetected, self.on_land),
            ("vehicle_command_ack", VehicleCommandAck, self.on_ack),
        ):
            self.create_subscription(cls, topic("out", name, cls), callback, qos)
        self.mode_pub = self.create_publisher(
            OffboardControlMode, topic("in", "offboard_control_mode", OffboardControlMode), 10
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, topic("in", "trajectory_setpoint", TrajectorySetpoint), 10
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, topic("in", "vehicle_command", VehicleCommand), 10
        )
        self.started = time.monotonic()
        self.phase = "waiting_for_telemetry"
        self.target: list[float] | None = None
        self.pre_roll = 0.0
        self.last_command = 0.0
        self.hold_since: float | None = None
        self.hold_errors: list[float] = []
        self.airborne = False
        self.done = False
        self.passed = False
        self.reason = ""
        self.land_started = 0.0
        self.create_timer(0.05, self.tick)

    def on_position(self, message: VehicleLocalPosition) -> None:
        self.position = message
        self.last_position = time.monotonic()

    def on_status(self, message: VehicleStatus) -> None:
        self.status = message
        self.last_status = time.monotonic()

    def on_land(self, message: VehicleLandDetected) -> None:
        self.landed = message
        self.last_land = time.monotonic()

    def on_ack(self, message: VehicleCommandAck) -> None:
        self.record({"command": int(message.command), "result": int(message.result)})

    def record(self, event: dict) -> None:
        record = {"elapsed_s": time.monotonic() - self.started, **event}
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()

    def command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        message = VehicleCommand()
        message.timestamp = self.get_clock().now().nanoseconds // 1000
        message.command = command
        message.param1 = param1
        message.param2 = param2
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 191
        message.from_external = True
        self.command_pub.publish(message)

    def start_landing(self, reason: str = "") -> None:
        self.reason = reason
        self.phase = "landing"
        self.land_started = time.monotonic()
        self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        print(f"Landing: {reason or 'hold complete'}", flush=True)

    def tick(self) -> None:
        now = time.monotonic()
        elapsed = now - self.started
        p, status = self.position, self.status
        fresh = now - self.last_position < 1 and now - self.last_status < 2
        if elapsed > 150 and self.phase != "landing":
            self.start_landing("150 s deadline exceeded")
        if self.phase == "landing":
            if now - self.last_command > 2:
                self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_command = now
            if (
                fresh
                and self.landed is not None
                and now - self.last_land < 2
                and self.landed.landed
                and status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self.passed = self.airborne and not self.reason
                self.done = True
            elif now - self.land_started > 45:
                self.reason = self.reason or "Landing/disarm confirmation timed out"
                self.done = True
        elif p is not None and status is not None and fresh and p.xy_valid and p.z_valid:
            if self.target is None:
                offset = enu_to_ned([0, 0, 5])
                self.target = [float(a + b) for a, b in zip(offset, (p.x, p.y, p.z), strict=True)]
                self.pre_roll = now
                self.phase = "pre_roll"
                self.record({"target_ned_m": self.target})
            mode = OffboardControlMode()
            mode.timestamp = self.get_clock().now().nanoseconds // 1000
            mode.position = True
            self.mode_pub.publish(mode)
            setpoint = TrajectorySetpoint()
            setpoint.timestamp = mode.timestamp
            setpoint.position = self.target
            setpoint.velocity = [math.nan] * 3
            setpoint.acceleration = [math.nan] * 3
            setpoint.yaw = enu_yaw_to_ned(math.pi / 2)  # Face north.
            self.setpoint_pub.publish(setpoint)
            if now - self.pre_roll > 3 and now - self.last_command > 2:
                if status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                    self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                elif status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                    self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.last_command = now
            error = math.dist([p.x, p.y, p.z], self.target)
            speed = math.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
            controlled = (
                status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                and status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            )
            if controlled and error < 0.5 and speed < 0.3:
                self.airborne = True
                if self.hold_since is None:
                    self.hold_since = now
                    self.hold_errors = []
                    self.phase = "holding"
                    print("Reached 5 m; holding for 10 s", flush=True)
                self.hold_errors.append(error)
                if now - self.hold_since >= 10:
                    self.start_landing()
            else:
                self.hold_since = None
        elif self.airborne and self.phase != "landing":
            self.start_landing("Telemetry became stale or invalid")
        if p is not None and status is not None:
            self.record(
                {
                    "phase": self.phase,
                    "position_ned_m": [p.x, p.y, p.z],
                    "velocity_ned_mps": [p.vx, p.vy, p.vz],
                    "nav_state": int(status.nav_state),
                    "arming_state": int(status.arming_state),
                    "landed": self.landed.landed if self.landed is not None else None,
                }
            )

    def finish(self) -> None:
        report = {
            "passed": self.passed,
            "reason": self.reason,
            "elapsed_s": time.monotonic() - self.started,
            "target_ned_m": self.target,
            "hold_s": 10 if self.passed else None,
            "hold_max_error_m": max(self.hold_errors, default=None),
            "landed_and_disarmed": self.passed,
        }
        (self.output / "flight-summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        self.stream.close()
        print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = FirstFlight(args.output)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()
    if not node.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
