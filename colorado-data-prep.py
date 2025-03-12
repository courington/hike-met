import os
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from shapely.geometry import box

# Define the area of interest (adjust coordinates for your specific area)
# These are approximate coordinates for Boulder to Golden area
xmin, ymin, xmax, ymax = -105.4, 39.7, -105.0, 40.1  # WGS84 coordinates

# Create a bounding box geometry
aoi_geom = box(xmin, ymin, xmax, ymax)
aoi = gpd.GeoDataFrame({'geometry': [aoi_geom]}, crs="EPSG:4326")

# Convert to UTM Zone 13N (appropriate for Colorado Front Range)
aoi_utm = aoi.to_crs("EPSG:26913")  # UTM Zone 13N

# Extract bounds in UTM coordinates
aoi_bounds_utm = aoi_utm.total_bounds

# Function to reproject and clip raster data
def process_raster(input_path, output_path, target_crs="EPSG:26913", resolution=10):
    """
    Reproject and clip raster to UTM Zone 13N at specified resolution
    
    Parameters:
    -----------
    input_path : str
        Path to input raster
    output_path : str
        Path for output raster
    target_crs : str
        Target coordinate reference system
    resolution : int
        Target resolution in meters
    """
    # Read the input raster
    with rasterio.open(input_path) as src:
        # Calculate the transformation parameters
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, 
            *src.bounds, resolution=(resolution, resolution)
        )
        
        # Update the metadata for the output raster
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create the output raster
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )
    
    # Now clip to AOI
    with rasterio.open(output_path) as src:
        window = rasterio.windows.from_bounds(
            aoi_bounds_utm[0], aoi_bounds_utm[1], 
            aoi_bounds_utm[2], aoi_bounds_utm[3],
            src.transform
        )
        
        # Read the data from the window
        data = src.read(window=window)
        
        # Update the transform for the window
        transform = rasterio.windows.transform(window, src.transform)
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': transform
        })
        
        # Create the clipped output
        clipped_output = output_path.replace('.tif', '_clipped.tif')
        with rasterio.open(clipped_output, 'w', **kwargs) as dst:
            dst.write(data)

# Function to process trails
def process_trails(input_path, output_path, target_crs="EPSG:26913"):
    """
    Reproject and clip trails to UTM Zone 13N
    
    Parameters:
    -----------
    input_path : str
        Path to input trails shapefile or geojson
    output_path : str
        Path for output trails file
    target_crs : str
        Target coordinate reference system
    """
    # Read trails data
    trails = gpd.read_file(input_path)
    
    # Reproject to target CRS
    trails_utm = trails.to_crs(target_crs)
    
    # Clip to AOI
    trails_clipped = gpd.clip(trails_utm, aoi_utm)
    
    # Save clipped trails
    trails_clipped.to_file(output_path)
    
    return trails_clipped

# Define input and output paths (update these with your actual file paths)
dem_input = "downloaded_data/ned_13_n40w106_1_arc_second.tif"
nlcd_input = "downloaded_data/nlcd_2019_land_cover_colorado.tif"
canopy_input = "downloaded_data/nlcd_2019_tree_canopy_colorado.tif"
snow_input = "downloaded_data/snodas_colorado_jan_avg.tif"  # January average
trails_input = "downloaded_data/colorado_trails.shp"

# Create output directory
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# Process each dataset
dem_output = os.path.join(output_dir, "dem_utm13n_10m.tif")
process_raster(dem_input, dem_output, resolution=10)  # Higher resolution for elevation

nlcd_output = os.path.join(output_dir, "landcover_utm13n_30m.tif")
process_raster(nlcd_input, nlcd_output, resolution=30)  # Original NLCD resolution

canopy_output = os.path.join(output_dir, "canopy_utm13n_30m.tif")
process_raster(canopy_input, canopy_output, resolution=30)

snow_output = os.path.join(output_dir, "snow_jan_utm13n_30m.tif")
process_raster(snow_input, snow_output, resolution=30)

trails_output = os.path.join(output_dir, "trails_utm13n.shp")
trails = process_trails(trails_input, trails_output)

print("Data preprocessing complete. All datasets are now in UTM Zone 13N projection.")
