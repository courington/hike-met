#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Remove existing venv if it exists
rm -rf venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
pip list

# Additional setup steps if needed...

# Install necessary packages
pip install geopandas rasterio numpy pandas matplotlib contextily flask folium requests arcgis2geojson ratelimit shapely branca