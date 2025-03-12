#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create and activate virtual environment in the script directory
python3 -m venv "$SCRIPT_DIR/hiking-map"
source "$SCRIPT_DIR/hiking-map/bin/activate"

# Install necessary packages
pip install geopandas rasterio numpy pandas matplotlib contextily flask folium requests arcgis2geojson ratelimit shapely branca