#!/bin/bash

set -xeo

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT/unit3"

git clone https://github.com/DLR-RM/rl-baselines3-zoo

cd "rl-baselines3-zoo"
pip install -r requirements.txt