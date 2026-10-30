#!/usr/bin/env bash
# Run as the normal Linux user. Sources/builds live on the WSL filesystem.
set -euo pipefail
[[ $EUID != 0 ]]
repo=$(cd "$(dirname "$0")/.." && pwd)
stage="$repo/data/m10/setup"
mkdir -p "$HOME/altiseek-m10" "$stage"
exec 9>"$HOME/altiseek-m10/build.lock"
flock -n 9 || { echo 'M10 build already running'; exit 1; }
echo $$ > "$stage/build.pid"
trap 'echo failed > "$stage/build-state"' ERR
echo fetching > "$stage/build-state"
cd "$HOME/altiseek-m10"

checkout() {
  local url=$1 ref=$2 commit=$3 dest=$4
  if [[ ! -d "$dest" ]]; then
    git clone --depth 1 --branch "$ref" "$url" "$dest"
  fi
  [[ $(git -C "$dest" rev-parse HEAD) == "$commit" ]]
  [[ -z $(git -C "$dest" status --porcelain --untracked-files=no) ]]
}
checkout https://github.com/PX4/PX4-Autopilot.git v1.17.0 \
  d6f12ad1c4f70ad3230afd7d86e971421e02fef4 PX4-Autopilot
git -C PX4-Autopilot submodule update --init --recursive --jobs 2
checkout https://github.com/eProsima/Micro-XRCE-DDS-Agent.git v2.4.3 \
  73622810d984349b80bbac0ef55fc0b694d62222 Micro-XRCE-DDS-Agent
mkdir -p ros2_ws/src
checkout https://github.com/PX4/px4_msgs.git release/1.17 \
  86d8239e962f6939e05c3737784f60c02fa884db ros2_ws/src/px4_msgs

echo waiting_for_system > "$stage/build-state"
while [[ $(cat "$stage/state") == running ]]; do sleep 10; done
[[ $(cat "$stage/state") == system_ready ]]
echo building > "$stage/build-state"
python3 -m venv --system-site-packages "$HOME/.venvs/px4-build"
source "$HOME/.venvs/px4-build/bin/activate"
python -m pip install -r PX4-Autopilot/Tools/setup/requirements.txt
python -m pip freeze > "$stage/px4-python-packages.txt"
make -C PX4-Autopilot -j2 px4_sitl_default
deactivate
cmake -S Micro-XRCE-DDS-Agent -B Micro-XRCE-DDS-Agent/build \
  -DCMAKE_INSTALL_PREFIX="$HOME/altiseek-m10/install"
cmake --build Micro-XRCE-DDS-Agent/build -j2
cmake --install Micro-XRCE-DDS-Agent/build
# ROS setup files can reference unset variables.
set +u
source /opt/ros/jazzy/setup.bash
set -u
cd ros2_ws
colcon build --packages-select px4_msgs --executor sequential
dpkg-query -W 'ros-jazzy-*' 'gz-*' 'libgz-*' > "$stage/system-packages.txt"
echo ready > "$stage/build-state"
printf '\nM10_BUILD_READY\n'
