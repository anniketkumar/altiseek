"""Use the unchanged simulator lifecycle with the D54 calibration adapter."""

from pathlib import Path

import run_camera_flight


def adapted_command(command: list[str]) -> list[str]:
    """Replace only the calibration controller, preserving every argument."""
    original = Path(__file__).resolve().with_name("calibration_flight.py")
    if len(command) > 1 and Path(command[1]) == original:
        return [command[0], str(original.with_name("calibration_recovery.py")), *command[2:]]
    return command


def main() -> None:
    original_popen = run_camera_flight.subprocess.Popen

    def launch(
        command: list[str], *args: object, **kwargs: object
    ) -> run_camera_flight.subprocess.Popen:
        return original_popen(adapted_command(command), *args, **kwargs)

    run_camera_flight.subprocess.Popen = launch
    try:
        run_camera_flight.main()
    finally:
        run_camera_flight.subprocess.Popen = original_popen


if __name__ == "__main__":
    main()
