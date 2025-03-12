import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import contextily as cx

def get_colorado_landcover_penalty(landcover_code):
    """
    Return speed penalty factor based on Colorado-specific NLCD landcover codes.
    
    Parameters:
    -----------
    landcover_code : int
        NLCD landcover classification code
    
    Returns:
    --------
    float
        Penalty factor (0.5-1.0) where 1.0 is no penalty
    """
    # NLCD specific code mapping
    landcover_penalties = {
        11: 0.9,    # Open Water
        12: 0.5,    # Perennial Ice/Snow
        21: 0.9,    # Developed, Open Space
        22: 0.8,    # Developed, Low Intensity
        23: 0.7,    # Developed, Medium Intensity
        24: 0.6,    # Developed, High Intensity
        31: 0.8,    # Barren Land
        41: 0.7,    # Deciduous Forest
        42: 0.7,    # Evergreen Forest
        43: 0.7,    # Mixed Forest
        52: 0.8,    # Shrub/Scrub (common in Front Range)
        71: 0.9,    # Grassland/Herbaceous (common in Front Range)
        81: 0.8,    # Pasture/Hay
        82: 0.7,    # Cultivated Crops
        90: 0.6,    # Woody Wetlands
        95: 0.5,    # Emergent Herbaceous Wetlands
    }
    
    # Default to moderate penalty if type not found
    return landcover_penalties.get(landcover_code, 0.7)

def get_front_range_snow_penalty(snow_depth_cm):
    """
    Return speed penalty factor based on snow depth for Front Range conditions.
    
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
    elif snow_depth_cm < 60:
        return 0.3
    else:
        return 0.2  # Deep snow in the Front Range can be very difficult

def calculate_colorado_hiking_metrics(trail_segments, dem_path, landcover_path, canopy_path, snow_path=None, hiker_weight=70):
    """
    Calculate hiking time and energy expenditure for Front Range trail segments.
    
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
    hiker_weight : float, optional
        Hiker weight in kg (default 70)
    
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
    results['elevation_gain_m'] = 0.0
    results['elevation_loss_m'] = 0.0
    results['max_elevation_m'] = 0.0
    results['avg_slope_pct'] = 0.0
    results['avg_canopy_pct'] = 0.0
    results['difficulty_score'] = 0.0
    
    # Open raster data sources
    with rasterio.open(dem_path) as dem, \
         rasterio.open(landcover_path) as landcover, \
         rasterio.open(canopy_path) as canopy:
        
        # Optional snow cover
        snow = None
        if snow_path and os.path.exists(snow_path):
            snow = rasterio.open(snow_path)
        
        # Process each trail segment
        for idx, segment in results.iterrows():
            # Sample points along the line (every 10 meters)
            line_length = segment.geometry.length
            num_points = max(20, int(line_length / 10))  # At least 20 points for short segments
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
            snow_profile = []
            if snow:
                snow_values = list(sample_gen(snow, coords))
                snow_profile = [v[0] if v[0] is not None else 0 for v in snow_values]
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
            
            # Calculate elevation metrics
            elevation_diffs = []
            elevation_gain = 0
            elevation_loss = 0
            
            for i in range(len(elevation_profile) - 1):
                elev_diff = elevation_profile[i+1] - elevation_profile[i]
                elevation_diffs.append(elev_diff)
                
                if elev_diff > 0:
                    elevation_gain += elev_diff
                else:
                    elevation_loss += abs(elev_diff)
            
            results.at[idx, 'elevation_gain_m'] = elevation_gain
            results.at[idx, 'elevation_loss_m'] = elevation_loss
            results.at[idx, 'max_elevation_m'] = max(elevation_profile)
            
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
            avg_canopy = np.mean([v for v in canopy_profile if v is not None]) if canopy_profile else 0
            results.at[idx, 'avg_canopy_pct'] = avg_canopy
            
            # Apply hiking speed and energy model
            time_minutes = 0
            calories = 0
            
            for i in range(len(slopes)):
                # Base walking speed using Tobler's function (km/h)
                # Modified for Front Range conditions
                slope = slopes[i]
                speed = 6 * np.exp(-3.5 * abs(slope + 0.05) / 100)
                
                # Front Range altitude penalty (reduce speed at higher elevations)
                altitude = elevation_profile[i]
                if altitude > 3000:  # 3000m ~ 9800ft
                    altitude_factor = 1.0 - min(0.2, (altitude - 3000) / 1000 * 0.1)
                else:
                    altitude_factor = 1.0
                
                # Apply terrain modifiers
                landcover_type = landcover_profile[i]
                landcover_factor = get_colorado_landcover_penalty(landcover_type)
                
                # Canopy penalty
                canopy_pct = canopy_profile[i] if canopy_profile[i] is not None else 0
                canopy_factor = 1.0 - (canopy_pct / 100 * 0.1)  # Less impact in Front Range
                
                # Snow penalty
                snow_depth = snow_profile[i] if i < len(snow_profile) else 0
                snow_factor = get_front_range_snow_penalty(snow_depth)
                
                # Apply all penalties to speed
                modified_speed = speed * landcover_factor * canopy_factor * snow_factor * altitude_factor
                
                # Calculate time for this segment (hours)
                segment_time_hours = (distances[i] / 1000) / modified_speed
                segment_time_minutes = segment_time_hours * 60
                time_minutes += segment_time_minutes
                
                # Calculate calories (MET-based approach)
                # Front Range altitude increases energy expenditure
                altitude_met_factor = 1.0 + max(0, (altitude - 2000) / 1000 * 0.1)
                
                # Base MET for hiking
                if abs(slope) < 3:
                    base_met = 3.5  # Walking, level ground
                elif abs(slope) < 6:
                    base_met = 5.3  # Hiking, hills
                elif abs(slope) < 10:
                    base_met = 6.0  # Hiking, moderate hills
                elif abs(slope) < 15:
                    base_met = 7.0  # Hiking, steep hills
                else:
                    base_met = 8.0  # Hiking, very steep hills
                
                # Adjust MET for slope direction
                if slope > 0:
                    # Uphill requires more energy
                    slope_adjustment = 1.0 + (slope / 100)
                else:
                    # Downhill can be more or less demanding depending on steepness
                    if abs(slope) < 10:
                        slope_adjustment = 0.9  # Slight downhill is easier
                    else:
                        slope_adjustment = 1.0 + (abs(slope) - 10) / 100  # Steep downhill is harder
                
                # Colorado-specific: Rocky terrain increases energy expenditure
                if landcover_type in [31, 52]:  # Barren or Shrub/Scrub often means rocky in Front Range
                    terrain_adjustment = 1.2
                else:
                    terrain_adjustment = 1.0
                
                # Adjust for snow
                snow_adjustment = 1.0 + (snow_depth / 100)
                
                # Calculate final MET
                final_met = base_met * slope_adjustment * terrain_adjustment * snow_adjustment * altitude_met_factor
                
                # Calculate calories (based on time and MET)
                segment_calories = (final_met * hiker_weight * 3.5 / 200) * segment_time_hours * 60
                calories += segment_calories
            
            # Store results
            results.at[idx, 'time_minutes'] = time_minutes
            results.at[idx, 'calories'] = calories
            
            # Calculate Front Range specific difficulty score (1-10)
            difficulty_score = (
                0.3 * min(avg_slope / 25, 1) +  # Slope contribution
                0.2 * min(total_distance_km / 8, 1) +  # Distance contribution
                0.2 * min(elevation_gain / 800, 1) +  # Elevation gain contribution (significant in Front Range)
                0.1 * min(results.at[idx, 'max_elevation_m'] / 3500, 1) + # High altitude factor
                0.1 * (avg_canopy / 100) +  # Canopy/forest factor
                0.1 * (np.mean(snow_profile) / 50 if snow_profile else 0)  # Snow factor
            )
            results.at[idx, 'difficulty_score'] = difficulty_score * 10  # Scale to 0-10
        
        # Close snow dataset if opened
        if snow:
            snow.close()
    
    return results

def create_colorado_hiking_map(trails_with_metrics, output_dir, dem_path=None, season="summer"):
    """
    Create maps visualizing hiking metrics for Colorado Front Range trails
    
    Parameters:
    -----------
    trails_with_metrics : GeoDataFrame
        Trail segments with calculated metrics
    output_dir : str
        Directory to save output maps
    dem_path : str, optional
        Path to DEM for hillshade background
    season : str
        Season name for map title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy of the trails for plotting
    plot_trails = trails_with_metrics.copy()
    
    # Convert to WGS84 for basemap
    plot_trails_wgs84 = plot_trails.to_crs(epsg=4326)
    
    # Create a custom colormap from green to red
    cmap = LinearSegmentedColormap.from_list("hiking_difficulty", 
                                             [(0, 'green'), (0.5, 'yellow'), (1, 'red')])
    
    # Plot time map
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_trails_wgs84.plot(column='time_minutes', ax=ax, linewidth=2, 
                          cmap=cmap, legend=True,
                          legend_kwds={'label': "Hiking Time (minutes)"})
    cx.add_basemap(ax, crs=plot_trails_wgs84.crs, source=cx.providers.Stamen.TerrainBackground)
    ax.set_title(f"Front Range Hiking Times - {season.capitalize()} Conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hiking_time_{season}.png"), dpi=300)
    
    # Plot calories map
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_trails_wgs84.plot(column='calories', ax=ax, linewidth=2, 
                          cmap=cmap, legend=True,
                          legend_kwds={'label': "Energy Expenditure (calories)"})
    cx.add_basemap(ax, crs=plot_trails_wgs84.crs, source=cx.providers.Stamen.TerrainBackground)
    ax.set_title(f"Front Range Energy Expenditure - {season.capitalize()} Conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hiking_energy_{season}.png"), dpi=300)
    
    # Plot difficulty map
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_trails_wgs84.plot(column='difficulty_score', ax=ax, linewidth=2, 
                          cmap=cmap, legend=True, 
                          legend_kwds={'label': "Difficulty (1-10)"})
    cx.add_basemap(ax, crs=plot_trails_wgs84.crs, source=cx.providers.Stamen.TerrainBackground)
    ax.set_title(f"Front Range Trail Difficulty - {season.capitalize()} Conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hiking_difficulty_{season}.png"), dpi=300)
    
    print(f"Maps created and saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Define paths to processed data
    processed_dir = "processed_data"
    dem_path = os.path.join(processed_dir, "dem_utm13n_10m.tif")
    landcover_path = os.path.join(processed_dir, "landcover_utm13n_30m.tif")
    canopy_path = os.path.join(processed_dir, "canopy_utm13n_30m.tif")
    
    # Seasonal snow data
    winter_snow_path = os.path.join(processed_dir, "snow_jan_utm13n_30m.tif")
    spring_snow_path = os.path.join(processed_dir, "snow_apr_utm13n_30m.tif")
    
    # Trails data
    trails_path = os.path.join(processed_dir, "trails_utm13n.shp")
    trails = gpd.read_file(trails_path)
    
    # Create output directory
    results_dir = "hiking_results"
    os.makedirs(results_dir, exist_ok=True)
    maps_dir = os.path.join(results_dir, "maps")
    
    # Process for summer conditions (no snow)
    print("Processing summer conditions...")
    summer_results = calculate_colorado_hiking_metrics(
        trails, 
        dem_path=dem_path,
        landcover_path=landcover_path,
        canopy_path=canopy_path
    )
    summer_results.to_file(os.path.join(results_dir, "summer_hiking_metrics.shp"))
    create_colorado_hiking_map(summer_results, maps_dir, dem_path, season="summer")
    
    # Process for winter conditions
    print("Processing winter conditions...")
    if os.path.exists(winter_snow_path):
        winter_results = calculate_colorado_hiking_metrics(
            trails, 
            dem_path=dem_path,
            landcover_path=landcover_path,
            canopy_path=canopy_path,
            snow_path=winter_snow_path
        )
        winter_results.to_file(os.path.join(results_dir, "winter_hiking_metrics.shp"))
        create_colorado_hiking_map(winter_results, maps_dir, dem_path, season="winter")
    
    print("Hiking analysis complete!")
