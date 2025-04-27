#!/bin/bash

# Create virtual environment
python3 -m venv venv

source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install torch torchvision matplotlib tqdm

pip freeze > requirements.txt

echo "Setup"