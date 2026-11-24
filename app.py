from flask import Flask, render_template, jsonify, request
import folium
import geopandas as gpd
import os
from arcgis2geojson import arcgis2geojson
import requests
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely import wkt
from pyproj import Geod
import branca.colormap as cm
# from colorado_hiking_model import calculate_colorado_hiking_metrics, get_colorado_landcover_penalty
# from colorado_interactive_map import create_interactive_hiking_map
from ratelimit import limits, sleep_and_retry
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from folium import plugins

# Set up logging. Set LOG_LEVEL=DEBUG to see request/response details.
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

# Rate limiting, shared by all outbound API calls. USGS doesn't publish a limit
# for EPQS; one request a second stays well under the 5-10/s other USGS
# services suggest. HTTP 429s are logged and not cached, so they're retried.
CALLS_PER_MINUTE = 60
RESPONSE_CACHE_SIZE = 1000

# Elevation lookups run in parallel because EPQS latency (often several seconds
# per point), not the rate limit, is what makes a trail slow. Every request still
# goes through the shared rate limiter.
ELEVATION_WORKERS = 5

# (connect, read) timeouts in seconds. EPQS often takes 14-17s to answer for a
# point it hasn't been asked about recently, so the read timeout needs headroom.
REQUEST_TIMEOUT = (5, 45)

# WGS84 ellipsoid, for measuring lon/lat geometries in meters
GEOD = Geod(ellps='WGS84')

# Successful responses only, keyed by (url, sorted params JSON), so a
# timeout or error is retried on the next request instead of being cached.
_response_cache = {}
_response_cache_lock = threading.Lock()

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
def _throttled_get(url, params):
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers={'Accept': 'application/json'})
    response.raise_for_status()
    return response.json()

def rate_limited_get(url, params=None):
    """Make a rate-limited, cached GET request. Returns parsed JSON or None on failure."""
    key = (url, json.dumps(params, sort_keys=True))
    with _response_cache_lock:
        if key in _response_cache:
            return _response_cache[key]

    try:
        data = _throttled_get(url, params)
    except requests.exceptions.Timeout:
        logger.error(f"Timeout accessing {url}")
        return None
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"Error accessing {url}: {e}")
        return None

    with _response_cache_lock:
        if len(_response_cache) >= RESPONSE_CACHE_SIZE:
            _response_cache.pop(next(iter(_response_cache)))  # drop the oldest entry
        _response_cache[key] = data
    return data

def _get_elevation(coord):
    """Elevation in meters for one (lon, lat) point, or None on failure."""
    lon, lat = coord  # GeoJSON order
    params = {
        'x': lon,
        'y': lat,
        'units': 'Meters',
    }
    data = rate_limited_get("https://epqs.nationalmap.gov/v1/json", params)
    logger.debug(f"Elevation at lon={lon}, lat={lat}: {data}")
    try:
        return float(data['value'])
    except (TypeError, KeyError, ValueError):
        logger.warning(f"Failed to get elevation for lon={lon}, lat={lat}")
        return None

def get_elevation_data(coords):
    """Get elevations from the USGS National Map API, in the same order as coords."""
    with ThreadPoolExecutor(max_workers=ELEVATION_WORKERS) as pool:
        return list(pool.map(_get_elevation, coords))

def get_landcover_data(coords):
    """Get landcover data from NLCD with rate limiting"""
    nlcd_url = "https://landcover.usgs.gov/arcgis/rest/services/NLCD_Land_Cover/MapServer/identify"
    landcover_values = []
    
    # Coordinates are (lon, lat), GeoJSON order
    for lon, lat in coords:
        params = {
            'geometry': json.dumps({'x': lon, 'y': lat}),
            'geometryType': 'esriGeometryPoint',
            'returnGeometry': False,
            'f': 'json'
        }
        
        data = rate_limited_get(nlcd_url, params)
        if data and data.get('results'):
            landcover_values.append(data['results'][0]['attributes']['Pixel Value'])
        else:
            landcover_values.append(None)
            logger.warning(f"Failed to get landcover for lon={lon}, lat={lat}")
        
        time.sleep(0.1)
    
    return landcover_values

def get_canopy_data(coords):
    """Get canopy cover data from USFS with rate limiting"""
    canopy_url = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_TreeCanopyCover_01/MapServer/identify"
    canopy_values = []
    
    # Coordinates are (lon, lat), GeoJSON order
    for lon, lat in coords:
        params = {
            'geometry': json.dumps({'x': lon, 'y': lat}),
            'geometryType': 'esriGeometryPoint',
            'returnGeometry': False,
            'f': 'json'
        }
        
        data = rate_limited_get(canopy_url, params)
        if data and data.get('results'):
            canopy_values.append(data['results'][0]['attributes']['Percent_Tree_Cover'])
        else:
            canopy_values.append(0)
            logger.warning(f"Failed to get canopy cover for lon={lon}, lat={lat}")
        
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

def calculate_trail_metrics(trail_id, geometry_wkt):
    """Calculate hiking metrics for a trail. Returns None if no elevations could be fetched.

    Not cached here: the underlying API responses are cached by rate_limited_get, and
    caching this would also cache a result built from failed elevation lookups.
    """
    try:
        geometry = LineString(wkt.loads(geometry_wkt))
        # Sample fewer points to stay within rate limits
        coords = sample_trail_points(geometry)
        logger.debug(f"Trail {trail_id}: sampled {len(coords)} points")

        # Get data from services
        elevations = get_elevation_data(coords)
        logger.debug(f"Trail {trail_id}: elevations {elevations}")
        valid_elevations = [e for e in elevations if e is not None]
        if not valid_elevations:
            logger.error(f"No elevation data for trail {trail_id}")
            return None

        # landcover = get_landcover_data(coords)
        # canopy = get_canopy_data(coords)

        # Geodesic length of the full trail (coords are lon/lat degrees), not just the sampled points
        total_distance_km = GEOD.geometry_length(geometry) / 1000
        logger.debug(f"Trail {trail_id}: total distance {total_distance_km} km")
        
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
        avg_elevation = float(np.mean(valid_elevations))
        altitude_factor = 1.0 + max(0, (avg_elevation - 2000) / 1000 * 0.1)
        calories = total_time_minutes * 7 * altitude_factor  # Rough estimate of calories per minute hiking

        # Difficulty score (0-10). The canopy term is 0 while get_canopy_data is
        # disabled, so scores currently top out at 8.
        canopy = []  # get_canopy_data(coords)
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
            'difficulty_score': float(difficulty_score),
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
        
        trails_data = rate_limited_get(trails_url, params)

        if not trails_data:
            return "Error loading trail data. Please try again later.", 503
        
        trails_gdf = gpd.GeoDataFrame.from_features(trails_data['features'])
        logger.debug(f"Loaded {len(trails_gdf)} trails")

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
                if (!response.ok || metrics.error) {
                    metricsDiv.innerHTML = 'Could not calculate metrics: ' + (metrics.error || response.statusText);
                    return;
                }

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