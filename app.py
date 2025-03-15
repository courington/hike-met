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
# from colorado_hiking_model import calculate_colorado_hiking_metrics, get_colorado_landcover_penalty
# from colorado_interactive_map import create_interactive_hiking_map
from ratelimit import limits, sleep_and_retry
from functools import lru_cache
import time
import logging
from folium import plugins

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting decorators
CALLS_PER_MINUTE = 6
RATE_LIMIT = 60 / CALLS_PER_MINUTE  # Seconds between calls

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
@lru_cache(maxsize=1000)
def rate_limited_get(url, params_str=None):
    """Make a rate-limited GET request with error handling and caching"""
    try:
        # Convert string back to dict if provided
        params = eval(params_str) if params_str else None
        
        # Set default headers
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        response = requests.get(url, params=params, timeout=10, headers=headers)
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
        print("lat", lat)
        print("lon", lon)
        params = {
            'x': lat,
            'y': lon,
            'units': 'Meters',
        }
        
        data = rate_limited_get(base_url, str(params))  # Remove headers parameter
        print("data", data)
        if data and 'value' in data:
            elevations.append(data['value'])
        else:
            elevations.append(None)
            logger.warning(f"Failed to get elevation for coordinates {lat}, {lon}")
        
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
        
        data = rate_limited_get(nlcd_url, str(params))  # Convert params to string
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
        
        data = rate_limited_get(canopy_url, str(params))  # Convert params to string
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
    print("coords 2", coords)
    if len(coords) <= max_points:
        return coords
    
    # Sample evenly spaced points
    indices = np.linspace(0, len(coords) - 1, max_points, dtype=int)
    print("indices", indices)
    return [coords[i] for i in indices]

@lru_cache(maxsize=100)
def calculate_trail_metrics(trail_id, geometry_wkt):
    """Calculate hiking metrics with caching based on trail ID"""
    try:
        geometry = LineString(wkt.loads(geometry_wkt))
        # Sample fewer points to stay within rate limits
        coords = sample_trail_points(geometry)
        print("coords", coords)

        # Get data from services
        elevations = get_elevation_data(coords)
        print("elevations", elevations)

        # landcover = get_landcover_data(coords)
        # canopy = get_canopy_data(coords)
        
        # Calculate basic metrics
        distances = []
        for i in range(len(coords) - 1):
            p1 = Point(coords[i])
            p2 = Point(coords[i + 1])
            distances.append(p1.distance(p2))
        
        total_distance_km = sum(distances) / 1000
        print("total_distance_km", total_distance_km)
        
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
        
        # # Calculate difficulty score
        # difficulty_score = (
        #     0.3 * min(elevation_gain / 800, 1) +
        #     0.3 * min(total_distance_km / 8, 1) +
        #     0.2 * min(avg_elevation / 3500, 1) +
        #     0.2 * (np.mean([c for c in canopy if c is not None] or [0]) / 100)
        # ) * 10

        return {
            'distance_km': total_distance_km,
            'elevation_gain_m': elevation_gain,
            'elevation_loss_m': elevation_loss,
            'time_minutes': total_time_minutes,
            'calories': calories,
            # 'difficulty_score': difficulty_score
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
            tiles='OpenStreetMap',  # Using OpenStreetMap as default for now
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under CC BY 4.0. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
        )
        print("m", m)
        # Add controls
        # folium.Scale().add_to(m)
        # plugins.Fullscreen().add_to(m)
        
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
        
        trails_data = rate_limited_get(
            trails_url, 
            str(params) if params else None
        )
        print("trails_data", trails_data)

        if not trails_data:
            return "Error loading trail data. Please try again later.", 503
        
        trails_gdf = gpd.GeoDataFrame.from_features(trails_data['features'])
        print("trails_gdf", trails_gdf)

        # # Create feature groups for different metrics
        # time_group = folium.FeatureGroup(name='Hiking Time')
        # calories_group = folium.FeatureGroup(name='Calories Burned')
        # difficulty_group = folium.FeatureGroup(name='Trail Difficulty')

        # # Create colormaps
        # time_colormap = cm.linear.YlOrRd_09.scale(0, 300)  # 0-5 hours
        # calories_colormap = cm.linear.YlOrRd_09.scale(0, 1000)  # 0-1000 calories
        # difficulty_colormap = cm.linear.YlOrRd_09.scale(0, 10)  # 0-10 difficulty

        # # Process trails with rate limiting
        # for idx, trail in trails_gdf.iterrows():
        #     metrics = calculate_trail_metrics(
        #         trail.get('OBJECTID', idx),
        #         trail.geometry.wkt
        #     )
            
        #     if metrics:
        #         # Create popup content
        #         popup_content = f"""
        #         <h4>{trail.get('name', f'Trail {idx}')}</h4>
        #         <b>Distance:</b> {metrics['distance_km']:.2f} km<br>
        #         <b>Elevation Gain:</b> {metrics['elevation_gain_m']:.1f} m<br>
        #         <b>Time:</b> {metrics['time_minutes']:.1f} minutes<br>
        #         <b>Calories:</b> {int(metrics['calories'])} cal<br>
        #         <b>Difficulty:</b> {metrics['difficulty_score']:.1f}/10
        #         """

        #         # Get line coordinates
        #         line_points = [(y, x) for x, y in zip(*trail.geometry.coords.xy)]

        #         # Add trails to each feature group
        #         for group, value, colormap in [
        #             (time_group, metrics['time_minutes'], time_colormap),
        #             (calories_group, metrics['calories'], calories_colormap),
        #             (difficulty_group, metrics['difficulty_score'], difficulty_colormap)
        #         ]:
        #             folium.PolyLine(
        #                 line_points,
        #                 color=colormap(value),
        #                 weight=3,
        #                 opacity=0.8,
        #                 popup=folium.Popup(popup_content, max_width=300)
        #             ).add_to(group)

        # Add all feature groups to map
        # time_group.add_to(m)
        # calories_group.add_to(m)
        # difficulty_group.add_to(m)

        # Add layer control
        # folium.LayerControl().add_to(m)

        # Add trails to map as polylines
        for idx, trail in trails_gdf.iterrows():
            # Create initial basic popup content
            initial_popup_content = f"""
            <h4>{trail.get('name', f'Trail {idx}')}</h4>
            <div id="metrics-{idx}">
                <button onclick="calculateMetrics('{idx}', '{trail.geometry.wkt}')">Calculate Trail Metrics</button>
            </div>
            """
            
            # Get line coordinates
            line_points = [(y, x) for x, y in zip(*trail.geometry.coords.xy)]
            
            # Add trail to map
            folium.PolyLine(
                line_points,
                color='blue',
                weight=2,
                opacity=0.8,
                popup=folium.Popup(initial_popup_content, max_width=300)
            ).add_to(m)

        # Add JavaScript to handle metric calculations
        js_code = """
        <script>
        async function calculateMetrics(trailId, geometryWkt) {
            const metricsDiv = document.getElementById(`metrics-${trailId}`);
            metricsDiv.innerHTML = 'Calculating...';
            
            try {
                const response = await fetch('/calculate_metrics', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        trail_id: trailId,
                        geometry_wkt: geometryWkt
                    })
                });
                
                const metrics = await response.json();
                
                metricsDiv.innerHTML = `
                    <b>Distance:</b> ${metrics.distance_km.toFixed(2)} km<br>
                    <b>Elevation Gain:</b> ${metrics.elevation_gain_m.toFixed(1)} m<br>
                    <b>Time:</b> ${metrics.time_minutes.toFixed(1)} minutes<br>
                    <b>Calories:</b> ${Math.round(metrics.calories)}<br>
                    <b>Difficulty:</b> ${metrics.difficulty_score.toFixed(1)}/10
                `;
            } catch (error) {
                metricsDiv.innerHTML = 'Error calculating metrics';
                console.error('Error:', error);
            }
        }
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(js_code))

        return m._repr_html_()
    
    except Exception as e:
        logger.error(f"Error generating map: {str(e)}")
        return "An error occurred generating the map. Please try again later.", 500

@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics_route():
    try:
        data = request.get_json()
        trail_id = data['trail_id']
        geometry_wkt = data['geometry_wkt']
        
        metrics = calculate_trail_metrics(trail_id, geometry_wkt)
        if metrics:
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Failed to calculate metrics'}), 500
            
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 