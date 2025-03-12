import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import rasterio
from rasterio.sample import sample_gen

def calculate_hiking_metrics(trail_segments, dem_path, landcover_path, canopy_path, snow_path=None):
    """
    Calculate hiking time and energy expenditure for trail segments.
    
    Parameters:
    -----------
    trail_segments : GeoDataFrame
        LineString geometries representing trail segments
    dem_path : str
        Path to Digital Elevation Model (DEM) raster
    landcover_path : str
        Path to land cover raster
    canopy_path : str
        Path to canopy cover raster
    snow_path : str, optional
        Path to snow cover raster
    
    Returns:
    --------
    GeoDataFrame
        Trail segments with added columns for time (minutes) and energy (calories)
    """
    # Create output GeoDataFrame
    results = trail_segments.copy()
    
    # Add columns for results
    results['time_minutes'] = 0.0
    results['calories'] = 0.0
    results['distance_km'] = 0.0
    results['avg_slope_pct'] = 0.0
    results['avg_canopy_pct'] = 0.0
    results['difficulty_score'] = 0.0
    
    # Open raster data sources
    with rasterio.open(dem_path) as dem, \
         rasterio.open(landcover_path) as landcover, \
         rasterio.open(canopy_path) as canopy:
        
        # Process each trail segment
        for idx, segment in results.iterrows():
            # Sample points along the line (every 10 meters)
            line_length = segment.geometry.length
            num_points = max(10, int(line_length / 10))  # At least 10 points
            segment_points = [segment.geometry.interpolate(i/float(num_points-1), normalized=True) 
                             for i in range(num_points)]
            
            # Extract coords for sampling
            coords = [(p.x, p.y) for p in segment_points]
            
            # Sample DEM for elevation
            elevation_values = list(sample_gen(dem, coords))
            elevation_profile = [v[0] for v in elevation_values]
            
            # Sample landcover
            landcover_values = list(sample_gen(landcover, coords))
            landcover_profile = [v[0] for v in landcover_values]
            
            # Sample canopy
            canopy_values = list(sample_gen(canopy, coords))
            canopy_profile = [v[0] for v in canopy_values]
            
            # Sample snow if provided
            if snow_path:
                with rasterio.open(snow_path) as snow:
                    snow_values = list(sample_gen(snow, coords))
                    snow_profile = [v[0] for v in snow_values]
            else:
                snow_profile = [0] * len(coords)  # Default to no snow
            
            # Calculate horizontal distance between points
            distances = []
            for i in range(len(segment_points) - 1):
                p1, p2 = segment_points[i], segment_points[i+1]
                distances.append(p1.distance(p2))
            
            # Calculate distance in km
            total_distance_km = sum(distances) / 1000
            results.at[idx, 'distance_km'] = total_distance_km
            
            # Calculate slopes
            slopes = []
            for i in range(len(elevation_profile) - 1):
                elev_diff = elevation_profile[i+1] - elevation_profile[i]
                if distances[i] > 0:
                    slope_pct = (elev_diff / distances[i]) * 100
                    slopes.append(slope_pct)
                else:
                    slopes.append(0)
            
            avg_slope = np.mean(np.abs(slopes)) if slopes else 0
            results.at[idx, 'avg_slope_pct'] = avg_slope
            
            # Calculate average canopy cover
            avg_canopy = np.mean(canopy_profile) if canopy_profile else 0
            results.at[idx, 'avg_canopy_pct'] = avg_canopy
            
            # Apply Tobler's hiking function with modifications for terrain factors
            time_minutes = 0
            calories = 0
            
            for i in range(len(slopes)):
                # Base walking speed using Tobler's function (km/h)
                slope = slopes[i]
                speed = 6 * np.exp(-3.5 * abs(slope + 0.05) / 100)
                
                # Apply terrain modifiers
                # Landcover penalty (0-0.5 reduction factor)
                landcover_type = landcover_profile[i]
                landcover_factor = get_landcover_penalty(landcover_type)
                
                # Canopy penalty (0-0.2 reduction factor)
                canopy_pct = canopy_profile[i]
                canopy_factor = 1.0 - (canopy_pct / 100 * 0.2)
                
                # Snow penalty (0-0.7 reduction factor)
                snow_depth = snow_profile[i]
                snow_factor = get_snow_penalty(snow_depth)
                
                # Apply all penalties to speed
                modified_speed = speed * landcover_factor * canopy_factor * snow_factor
                
                # Calculate time for this segment (hours)
                segment_time_hours = (distances[i] / 1000) / modified_speed
                segment_time_minutes = segment_time_hours * 60
                time_minutes += segment_time_minutes
                
                # Calculate calories (MET-based approach)
                weight_kg = 70  # Default weight, should be parameterized
                met_value = get_hiking_met(slope, landcover_type, snow_depth)
                segment_calories = (met_value * weight_kg * 3.5 / 200) * segment_time_hours * 60
                calories += segment_calories
            
            # Store results
            results.at[idx, 'time_minutes'] = time_minutes
            results.at[idx, 'calories'] = calories
            
            # Calculate difficulty score (simple weighted average)
            difficulty_score = (
                0.4 * min(avg_slope / 30, 1) +  # Max slope contribution at 30%
                0.3 * min(total_distance_km / 10, 1) +  # Max distance contribution at 10km
                0.2 * (avg_canopy / 100) +
                0.1 * (np.mean(snow_profile) / 50 if snow_profile else 0)  # Max snow contribution at 50cm
            )
            results.at[idx, 'difficulty_score'] = difficulty_score * 10  # Scale to 0-10
    
    return results

def get_landcover_penalty(landcover_type):
    """
    Return speed penalty factor based on landcover type.
    
    Parameters:
    -----------
    landcover_type : int
        Landcover classification code
    
    Returns:
    --------
    float
        Penalty factor (0.5-1.0) where 1.0 is no penalty
    """
    # Example mapping - should be calibrated to your landcover classification
    landcover_penalties = {
        1: 1.0,    # Bare ground / trail
        2: 0.9,    # Grass / meadow
        3: 0.8,    # Shrubs / light vegetation
        4: 0.7,    # Forest
        5: 0.6,    # Dense vegetation
        6: 0.5,    # Very difficult terrain (marsh, boulder field)
    }
    
    # Default to moderate penalty if type not found
    return landcover_penalties.get(landcover_type, 0.7)

def get_snow_penalty(snow_depth_cm):
    """
    Return speed penalty factor based on snow depth.
    
    Parameters:
    -----------
    snow_depth_cm : float
        Snow depth in centimeters
    
    Returns:
    --------
    float
        Penalty factor (0.3-1.0) where 1.0 is no penalty
    """
    if snow_depth_cm <= 0:
        return 1.0
    elif snow_depth_cm < 5:
        return 0.9
    elif snow_depth_cm < 15:
        return 0.7
    elif snow_depth_cm < 30:
        return 0.5
    else:
        return 0.3

def get_hiking_met(slope_pct, landcover_type, snow_depth_cm):
    """
    Calculate MET (Metabolic Equivalent of Task) value for hiking.
    
    Parameters:
    -----------
    slope_pct : float
        Slope as percentage
    landcover_type : int
        Landcover classification code
    snow_depth_cm : float
        Snow depth in centimeters
    
    Returns:
    --------
    float
        MET value for hiking under these conditions
    """
    # Base MET for hiking
    if abs(slope_pct) < 3:
        base_met = 3.5  # Walking, level ground
    elif abs(slope_pct) < 6:
        base_met = 5.3  # Hiking, hills
    elif abs(slope_pct) < 10:
        base_met = 6.0  # Hiking, moderate hills
    elif abs(slope_pct) < 15:
        base_met = 7.0  # Hiking, steep hills
    else:
        base_met = 8.0  # Hiking, very steep hills
    
    # Adjust MET for uphill vs downhill
    if slope_pct > 0:
        # Uphill requires more energy
        slope_adjustment = 1.0 + (slope_pct / 100)
    else:
        # Downhill can be more or less demanding depending on steepness
        if abs(slope_pct) < 10:
            slope_adjustment = 0.9  # Slight downhill is easier
        else:
            slope_adjustment = 1.0 + (abs(slope_pct) - 10) / 100  # Steep downhill is harder
    
    # Adjust for landcover
    landcover_adjustment = 1.0 + ((6 - get_landcover_penalty(landcover_type) * 10) / 10)
    
    # Adjust for snow
    if snow_depth_cm <= 0:
        snow_adjustment = 1.0
    elif snow_depth_cm < 5:
        snow_adjustment = 1.1
    elif snow_depth_cm < 15:
        snow_adjustment = 1.3
    elif snow_depth_cm < 30:
        snow_adjustment = 1.6
    else:
        snow_adjustment = 2.0
    
    # Calculate final MET
    final_met = base_met * slope_adjustment * landcover_adjustment * snow_adjustment
    
    return final_met

# Example usage:
# trail_segments = gpd.read_file("trail_network.shp")
# results = calculate_hiking_metrics(
#     trail_segments,
#     "elevation.tif",
#     "landcover.tif",
#     "canopy.tif",
#     "snow_january.tif"
# )
# results.to_file("hiking_metrics.shp")
