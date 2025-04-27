#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install torch torchvision matplotlib tqdm

# Optional: freeze requirements
pip freeze > requirements.txt

echo "Setup"