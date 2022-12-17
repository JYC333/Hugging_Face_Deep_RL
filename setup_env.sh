#!/bin/bash

set -xeo

if [ ! -d "/venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

pip install stable-baselines3[extra]
pip install gymnasium[all]
pip install box2d box2d-kengz huggingface_sb3 pygame
pip install pyglet==1.5.27
pip install box2d-py==2.3.8