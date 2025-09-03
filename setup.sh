#!/bin/bash
# setup.sh - Run this once to set up the environment

module load python/3.10

# Create virtual environment if it doesn't exist
if [ ! -d "pyROMenv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv pyROMenv
fi

source pyROMenv/bin/activate

# Install/update packages
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Virtual environment is ready."
echo "Installed packages:"
pip list

deactivate