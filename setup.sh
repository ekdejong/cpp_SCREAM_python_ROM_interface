#!/bin/bash
# setup.sh - Run this once to set up the environment

module load python/3.9

# Create virtual environment if it doesn't exist
if [ ! -d "env39" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env39
fi

source env39/bin/activate
which pip3 # verify that the correct pip installation is being used

# Install/update packages
echo "Installing required packages..."
pip install --upgrade pip
# You'll want to find the correct version of pytorch for your compute platform here:
# https://pytorch.org/get-started/locally/#start-locally
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scipy, matploblib, pybind11

echo "Setup complete! Virtual environment is ready."
echo "Installed packages:"
pip list

deactivate