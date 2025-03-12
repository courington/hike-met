from flask import Flask, render_template, jsonify, request
import folium
import geopandas as gpd
import os
from arcgis2geojson import arcgis2geojson
import requests
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely import wkt
import branca.colormap as cm
from colorado_hiking_model import calculate_colorado_hiking_metrics, get_colorado_landcover_penalty
from colorado_interactive_map import create_interactive_hiking_map
from ratelimit import limits, sleep_and_retry
from functools import lru_cache
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting decorators
CALLS_PER_MINUTE = 6
RATE_LIMIT = 60 / CALLS_PER_MINUTE  # Seconds between calls

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
@lru_cache(maxsize=1000)  # Cache results to avoid repeated API calls
def rate_limited_get(url, params=None):
    """Make a rate-limited GET request with error handling and caching"""
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Timeout accessing {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error accessing {url}: {str(e)}")
        return None

def get_elevation_data(coords):
    """Get elevation data from USGS National Map API with rate limiting"""
    base_url = "https://epqs.nationalmap.gov/v1/json"
    elevations = []
    
    for lat, lon in coords:
        params = {
            'x': lon,
            'y': lat,
            'units': 'Meters',
            'output': 'json'
        }
        
        data = rate_limited_get(base_url, params)
        if data and 'value' in data:
            elevations.append(data['value'])
        else:
            elevations.append(None)
            logger.warning(f"Failed to get elevation for coordinates {lat}, {lon}")
        
        # Add small delay between requests
        time.sleep(0.1)
    
    return elevations

def get_landcover_data(coords):
    """Get landcover data from NLCD with rate limiting"""
    nlcd_url = "https://landcover.usgs.gov/arcgis/rest/services/NLCD_Land_Cover/MapServer/identify"
    landcover_values = []
    
    for lat, lon in coords:
        params = {
            'geometry': {'x': lon, 'y': lat},
            'geometryType': 'esriGeometryPoint',
            'returnGeometry': False,
            'f': 'json'
        }
        
        data = rate_limited_get(nlcd_url, params)
        if data and data.get('results'):
            landcover_values.append(data['results'][0]['attributes']['Pixel Value'])
        else:
            landcover_values.append(None)
            logger.warning(f"Failed to get landcover for coordinates {lat}, {lon}")
        
        time.sleep(0.1)
    
    return landcover_values

def get_canopy_data(coords):
    """Get canopy cover data from USFS with rate limiting"""
    canopy_url = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_TreeCanopyCover_01/MapServer/identify"
    canopy_values = []
    
    for lat, lon in coords:
        params = {
            'geometry': {'x': lon, 'y': lat},
            'geometryType': 'esriGeometryPoint',
            'returnGeometry': False,
            'f': 'json'
        }
        
        data = rate_limited_get(canopy_url, params)
        if data and data.get('results'):
            canopy_values.append(data['results'][0]['attributes']['Percent_Tree_Cover'])
        else:
            canopy_values.append(0)
            logger.warning(f"Failed to get canopy cover for coordinates {lat}, {lon}")
        
        time.sleep(0.1)
    
    return canopy_values

def sample_trail_points(geometry, max_points=20):
    """Sample a limited number of points along the trail to reduce API calls"""
    coords = list(geometry.coords)
    if len(coords) <= max_points:
        return coords
    
    # Sample evenly spaced points
    indices = np.linspace(0, len(coords) - 1, max_points, dtype=int)
    return [coords[i] for i in indices]

@lru_cache(maxsize=100)
def calculate_trail_metrics(trail_id, geometry_wkt):
    """Calculate hiking metrics with caching based on trail ID"""
    try:
        geometry = LineString(wkt.loads(geometry_wkt))
        # Sample fewer points to stay within rate limits
        coords = sample_trail_points(geometry)
        
        # Get data from services
        elevations = get_elevation_data(coords)
        landcover = get_landcover_data(coords)
        canopy = get_canopy_data(coords)
        
        # Calculate basic metrics
        distances = []
        for i in range(len(coords) - 1):
            p1 = Point(coords[i])
            p2 = Point(coords[i + 1])
            distances.append(p1.distance(p2))
        
        total_distance_km = sum(distances) / 1000
        
        # Calculate elevation metrics
        elevation_gain = 0
        elevation_loss = 0
        for i in range(len(elevations) - 1):
            if elevations[i] is not None and elevations[i+1] is not None:
                diff = elevations[i+1] - elevations[i]
                if diff > 0:
                    elevation_gain += diff
                else:
                    elevation_loss += abs(diff)
        
        # Calculate hiking time (Naismith's Rule with modifications)
        base_time_hours = total_distance_km / 5.0
        ascent_time = elevation_gain / 600.0
        descent_time = elevation_loss / 1200.0
        total_time_minutes = (base_time_hours + ascent_time + descent_time) * 60
        
        # Calculate calories (basic estimation)
        avg_elevation = np.mean([e for e in elevations if e is not None])
        altitude_factor = 1.0 + max(0, (avg_elevation - 2000) / 1000 * 0.1)
        calories = total_time_minutes * 7 * altitude_factor  # Rough estimate of calories per minute hiking
        
        # Calculate difficulty score
        difficulty_score = (
            0.3 * min(elevation_gain / 800, 1) +
            0.3 * min(total_distance_km / 8, 1) +
            0.2 * min(avg_elevation / 3500, 1) +
            0.2 * (np.mean([c for c in canopy if c is not None] or [0]) / 100)
        ) * 10

        return {
            'distance_km': total_distance_km,
            'elevation_gain_m': elevation_gain,
            'elevation_loss_m': elevation_loss,
            'time_minutes': total_time_minutes,
            'calories': calories,
            'difficulty_score': difficulty_score
        }
    except Exception as e:
        logger.error(f"Error calculating metrics for trail {trail_id}: {str(e)}")
        return None

app = Flask(__name__)

@app.route('/')
def show_map():
    try:
        # Create base map
        m = folium.Map(
            location=[39.113014, -105.358887],
            zoom_start=7,
            tiles='Stamen Terrain',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
        )
        
        # Add controls
        folium.ScaleControl(imperial=True, metric=True).add_to(m)
        folium.plugins.Fullscreen().add_to(m)
        
        # Get trails data from COTREX with rate limiting
        trails_url = "https://services1.arcgis.com/82YxYqy3f0s2D9c4/arcgis/rest/services/Trails_COTREX02172021/FeatureServer/0/query"
        params = {
            'where': '1=1',
            'outFields': '*',
            'geometryType': 'esriGeometryEnvelope',
            'spatialRel': 'esriSpatialRelIntersects',
            'outSR': '4326',
            'f': 'geojson'
        }
        
        trails_data = rate_limited_get(trails_url, params)
        if not trails_data:
            return "Error loading trail data. Please try again later.", 503
        
        trails_gdf = gpd.GeoDataFrame.from_features(trails_data['features'])
        
        # Create feature groups for different metrics
        time_group = folium.FeatureGroup(name='Hiking Time')
        calories_group = folium.FeatureGroup(name='Calories Burned')
        difficulty_group = folium.FeatureGroup(name='Trail Difficulty')

        # Create colormaps
        time_colormap = cm.linear.YlOrRd_09.scale(0, 300)  # 0-5 hours
        calories_colormap = cm.linear.YlOrRd_09.scale(0, 1000)  # 0-1000 calories
        difficulty_colormap = cm.linear.YlOrRd_09.scale(0, 10)  # 0-10 difficulty

        # Process trails with rate limiting
        for idx, trail in trails_gdf.iterrows():
            metrics = calculate_trail_metrics(
                trail.get('OBJECTID', idx),
                trail.geometry.wkt
            )
            
            if metrics:
                # Create popup content
                popup_content = f"""
                <h4>{trail.get('name', f'Trail {idx}')}</h4>
                <b>Distance:</b> {metrics['distance_km']:.2f} km<br>
                <b>Elevation Gain:</b> {metrics['elevation_gain_m']:.1f} m<br>
                <b>Time:</b> {metrics['time_minutes']:.1f} minutes<br>
                <b>Calories:</b> {int(metrics['calories'])} cal<br>
                <b>Difficulty:</b> {metrics['difficulty_score']:.1f}/10
                """

                # Get line coordinates
                line_points = [(y, x) for x, y in zip(*trail.geometry.coords.xy)]

                # Add trails to each feature group
                for group, value, colormap in [
                    (time_group, metrics['time_minutes'], time_colormap),
                    (calories_group, metrics['calories'], calories_colormap),
                    (difficulty_group, metrics['difficulty_score'], difficulty_colormap)
                ]:
                    folium.PolyLine(
                        line_points,
                        color=colormap(value),
                        weight=3,
                        opacity=0.8,
                        popup=folium.Popup(popup_content, max_width=300)
                    ).add_to(group)

        # Add all feature groups to map
        time_group.add_to(m)
        calories_group.add_to(m)
        difficulty_group.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        return m._repr_html_()
    
    except Exception as e:
        logger.error(f"Error generating map: {str(e)}")
        return "An error occurred generating the map. Please try again later.", 500

if __name__ == '__main__':
    app.run(debug=True) 