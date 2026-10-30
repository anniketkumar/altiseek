"""Local SITL ground-station heartbeat; no arming or flight commands."""

import time

from pymavlink import mavutil


def main() -> None:
    link = mavutil.mavlink_connection(
        "udpout:127.0.0.1:18570", source_system=255, source_component=190
    )
    try:
        for _ in range(210):
            link.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE,
            )
            time.sleep(1)
    finally:
        link.close()


if __name__ == "__main__":
    main()
