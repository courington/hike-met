import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import os
import branca.colormap as cm

def create_interactive_hiking_map(trails_with_metrics, output_path, center_coords=None):
    """
    Create an interactive Folium map of hiking trails with metrics
    
    Parameters:
    -----------
    trails_with_metrics : GeoDataFrame
        Trail segments with calculated metrics
    output_path : str
        Path to save HTML output map
    center_coords : tuple, optional
        (lat, lon) coordinates to center the map
    """
    # Convert to WGS84 for web mapping
    trails_wgs84 = trails_with_metrics.to_crs(epsg=4326)
    
    # If center coordinates not provided, use centroid of all trails
    if not center_coords:
        all_trails = trails_wgs84.unary_union
        center_lat, center_lon = all_trails.centroid.y, all_trails.centroid.x
    else:
        center_lat, center_lon = center_coords
    
    # Create map centered on the area of interest
    map_center = [center_lat, center_lon]
    mymap = folium.Map(location=map_center, zoom_start=11, 
                      tiles='Stamen Terrain')
    
    # Add scale
    folium.ScaleControl(imperial=True, metric=True).add_to(mymap)
    
    # Add fullscreen option
    folium.plugins.Fullscreen().add_to(mymap)
    
    # Create colormaps for different metrics
    time_colormap = cm.linear.YlOrRd_09.scale(
        trails_wgs84['time_minutes'].min(),
        trails_wgs84['time_minutes'].max()
    )
    
    calories_colormap = cm.linear.YlOrRd_09.scale(
        trails_wgs84['calories'].min(),
        trails_wgs84['calories'].max()
    )
    
    difficulty_colormap = cm.linear.YlOrRd_09.scale(0, 10)
    
    # Create Feature Groups for each metric type (for layer control)
    time_group = folium.FeatureGroup(name='Hiking Time')
    calories_group = folium.FeatureGroup(name='Calories Burned')
    difficulty_group = folium.FeatureGroup(name='Trail Difficulty')
    
    # Add trail segments to the map
    for idx, trail in trails_wgs84.iterrows():
        # Format popup content
        popup_content = f"""
        <h4>{trail.get('name', f'Trail Segment {idx}')}</h4>
        <b>Distance:</b> {trail['distance_km']:.2f} km<br>
        <b>Elevation Gain:</b> {trail['elevation_gain_m']:.1f} m<br>
        <b>Max Elevation:</b> {trail['max_elevation_m']:.1f} m<br>
        <b>Time:</b> {trail['time_minutes']:.1f} minutes<br>
        <b>Calories:</b> {int(trail['calories'])} cal<br>
        <b>Difficulty:</b> {trail['difficulty_score']:.1f}/10<br>
        """
        
        # Get line coordinates
        line_points = [(y, x) for x, y in zip(*trail.geometry.coords.xy)]
        
        # Add time layer
        time_color = time_colormap(trail['time_minutes'])
        folium.PolyLine(
            line_points, 
            color=time_color,
            weight=5,
            opacity=0.8,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(time_group)
        
        # Add calories layer
        calories_color = calories_colormap(trail['calories'])
        folium.PolyLine(
            line_points,
            color=calories_color,
            weight=5,
            opacity=0.8,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(calories_group)
        
        # Add difficulty layer
        difficulty_color = difficulty_colormap(trail['difficulty_score'])
        folium.PolyLine(
            line_points,
            color=difficulty_color,
            weight=5,
            opacity=0.8,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(difficulty_group)
