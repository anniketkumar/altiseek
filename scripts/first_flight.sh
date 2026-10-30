#!/usr/bin/env bash
# One headless SITL flight only; does not start search or transfer experiments.
set -eo pipefail
cd "$(dirname "$0")/.."
source /opt/ros/jazzy/setup.bash
source /home/aniket/altiseek-m10/ros2_ws/install/setup.bash
exec /usr/bin/python3 scripts/run_first_flight.py
