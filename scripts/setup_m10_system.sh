#!/usr/bin/env bash
# Ubuntu 24.04 system dependencies; run as root inside the project's WSL distro.
# Based on PX4 v1.17.0 Tools/setup/ubuntu.sh, simulation only. Python build
# dependencies go in a separate user venv, never into the system interpreter.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
source /etc/os-release
[[ $EUID == 0 && $ID == ubuntu && $VERSION_ID == 24.04 ]]
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg locales
curl --fail --location --retry 3 \
  https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.3.0/ros2-apt-source_1.3.0.noble_all.deb \
  -o /tmp/altiseek-ros2-apt-source.deb
dpkg -i /tmp/altiseek-ros2-apt-source.deb
curl --fail --location --retry 3 https://packages.osrfoundation.org/gazebo.gpg \
  -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable noble main' \
  > /etc/apt/sources.list.d/altiseek-gazebo.list
apt-get update
apt-get install -y --no-install-recommends \
  ros-jazzy-ros-base ros-jazzy-ros-gz ros-jazzy-cv-bridge ros-dev-tools \
  astyle build-essential ccache cmake cppcheck file gdb git lcov libssl-dev \
  libxml2-dev libxml2-utils ninja-build python3-dev python3-venv python3-pip \
  python3-setuptools python3-wheel rsync shellcheck unzip zip bc \
  gz-harmonic libunwind-dev cppzmq-dev libeigen3-dev \
  libopencv-dev pkg-config protobuf-compiler mesa-utils
printf '\nM10_SYSTEM_READY\n'
