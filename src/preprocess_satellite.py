import os
import re

import random
from datetime import datetime
from typing import Dict, List, Union, Tuple
from cloudpathlib import S3Path, S3Client
from pathlib import Path
import geopandas as gpd
import folium

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC, SDS
from pyproj import CRS, Proj

# NEEDED FOR THE GENERALIZING PART
wgs84_crs = CRS.from_epsg("4326")

def calibrate_data(dataset: SDS, shape: List[int], calibration_dict: Dict):
    """Given a MAIAC dataset and calibration parameters, return a masked
    array of calibrated data.
    
    Args:
        dataset (SDS): dataset in SDS format (e.g. blue band AOD).
        shape (List[int]): dataset shape as a list of [orbits, height, width].
        calibration_dict (Dict): dictionary containing, at a minimum,
            `valid_range` (list or tuple), `_FillValue` (int or float),
            `add_offset` (float), and `scale_factor` (float).
    
    Returns:
        corrected_AOD (np.ma.MaskedArray): masked array of calibrated data
            with a fill value of nan.
    """
    corrected_AOD = np.ma.empty(shape, dtype=np.double)
    for orbit in range(shape[0]):
        data = dataset[orbit, :, :].astype(np.double)
        
        invalid_condition = ( # Create a True/False mask
            (data < calibration_dict["valid_range"][0]) |
            (data > calibration_dict["valid_range"][1]) |
            (data == calibration_dict["_FillValue"])
        )
        data[invalid_condition] = np.nan # Apply the T/F mask to data
        
        data = (
            (data - calibration_dict["add_offset"]) *
            calibration_dict["scale_factor"]
        )

        data = np.ma.masked_array(data, np.isnan(data)) # nan values get changed to "--"

        corrected_AOD[orbit, : :] = data
    
    corrected_AOD.fill_value = np.nan #Default value of a mask value: 1e20, change it to nan

    return corrected_AOD


def create_calibration_dict(data: SDS):
    """Define calibration dictionary given a SDS dataset,
    which contains:
        - name
        - scale factor
        - offset
        - unit
        - fill value
        - valid range
    
    Args:
        data (SDS): dataset in the SDS format.
    
    Returns:
        calibration_dict (Dict): dict of calibration parameters.
    """
    return data.attributes()


def create_alignment_dict(hdf: SD):
    """Define alignment dictionary given a SD data file, 
    which contains:
        - upper left coordinates
        - lower right coordinates
        - coordinate reference system (CRS)
        - CRS parameters
    
    Args:
        hdf (SD): hdf data object
    
    Returns:
        alignment_dict (Dict): dict of alignment parameters.
    """
    group_1 = hdf.attributes()["StructMetadata.0"].split("END_GROUP=GRID_1")[0]
    hdf_metadata = dict([x.split("=") for x in group_1.split() if "=" in x])

    alignment_dict = {
        "upper_left": eval(hdf_metadata["UpperLeftPointMtrs"]),
        "lower_right": eval(hdf_metadata["LowerRightMtrs"]),
        "crs": hdf_metadata["Projection"],
        "crs_params": eval(hdf_metadata["ProjParams"])
    }
    return alignment_dict


def create_meshgrid(alignment_dict: Dict, shape: List[int]):
    """Given an image shape, create a meshgrid of points
    between bounding coordinates.
    
    Args:
        alignment_dict (Dict): dictionary containing, at a minimum,
            `upper_left` (tuple), `lower_right` (tuple), `crs` (str),
            and `crs_params` (tuple).
        shape (List[int]): dataset shape as a list of
            [orbits, height, width].
    
    Returns:
        xv (np.array): x (longitude) coordinates.
        yv (np.array): y (latitude) coordinates.
    """
    # Determine grid bounds using two coordinates
    x0, y0 = alignment_dict["upper_left"]
    x1, y1 = alignment_dict["lower_right"]
    
    # print(f'upper-left: {alignment_dict["upper_left"] }')

    # print(f'lower-right: { alignment_dict["lower_right"] }')

    # Interpolate points between corners, inclusive of bounds
    x = np.linspace(x0, x1, shape[2], endpoint=True)
    y = np.linspace(y0, y1, shape[1], endpoint=True)
    
    # Return two 2D arrays representing X & Y coordinates of all points
    xv, yv = np.meshgrid(x, y)
    return xv, yv


def transform_arrays(
    xv: Union[np.array, float],
    yv: Union[np.array, float],
    crs_from: CRS,
    crs_to: CRS
):
    """Transform points or arrays from one CRS to another CRS.
    
    Args:
        xv (np.array or float): x (longitude) coordinates or value.
        yv (np.array or float): y (latitude) coordinates or value.
        crs_from (CRS): source coordinate reference system.
        crs_to (CRS): destination coordinate reference system.
    
    Returns:
        lon, lat (tuple): x coordinate(s), y coordinate(s)
    """
    transformer = pyproj.Transformer.from_crs(
        crs_from,
        crs_to,
        always_xy=True,
    )
    lon, lat = transformer.transform(xv, yv)
    return lon, lat


def convert_array_to_df(
    corrected_arr: np.ma.MaskedArray,
    lat:np.ndarray,
    lon: np.ndarray,
    granule_id: str,
    crs: CRS,
    total_bounds: np.ndarray = []
):
    """Align data values with latitude and longitude coordinates
    and return a GeoDataFrame.
    
    Args:
        corrected_arr (np.ma.MaskedArray): data values for each pixel.
        lat (np.ndarray): latitude for each pixel.
        lon (np.ndarray): longitude for each pixel.
        granule_id (str): granule name.
        crs (CRS): coordinate reference system
        total_bounds (np.ndarray, optional): If provided, will filter out points that fall
            outside of these bounds. Composed of xmin, ymin, xmax, ymax.
    """
    lats = lat.ravel()
    lons = lon.ravel()
    n_orbits = len(corrected_arr) # corrected_arr.shape : (num_orbits, 1200, 1200)
    size = lats.size
    values = {
        "value": np.concatenate([d.data.ravel() for d in corrected_arr]), # each d = one orbit
        "lat": np.tile(lats, n_orbits), #np.tile: repeat lats n_orbits number of times and concat in a 1D array
        "lon": np.tile(lons, n_orbits),
        "orbit": np.arange(n_orbits).repeat(size), # [0,0,0... ,1,1, ... 2,2,... n_orbits]
        "granule_id": [granule_id] * size * n_orbits
        
    }
    
    df = pd.DataFrame(values).dropna()

    if len(total_bounds) > 0: # total_bounds es array o df. Si len() > 0: significa que no es None

        if type(total_bounds) == list:
            lon_min, lat_min, lon_max, lat_max = total_bounds # Si el valor es una lista

        else: 
            lon_min, lat_min, lon_max, lat_max = total_bounds['geometry'].iloc[0].bounds # Si el valor es un gdf con un Polygon

        df = df[ df.lat.between(lat_min, lat_max) & df.lon.between(lon_min, lon_max)] 


    gdf = gpd.GeoDataFrame(df)
    gdf["geometry"] = gpd.points_from_xy(gdf.lon, gdf.lat)
    gdf.crs = crs
    
    return gdf[["granule_id", "orbit", "geometry", "value"]].reset_index(drop=True)


def get_granule(upper_left):
    if upper_left == -10007554.677:
        return 'R'
    else:
        return 'L'


def str_datetime(date_time_str):
    if date_time_str[-1] in ['T', 'A']:
        date_time_str = date_time_str[:-1]
    return datetime.strptime(date_time_str, '%Y%j%H%M')


def get_time_stamp(hdf):
    stripped = hdf.attributes()['Orbit_time_stamp'].strip()
    splitted = stripped.split()
    if len(splitted) > 1:
        start_time = str_datetime(splitted[0])
        end_time = str_datetime(splitted[1])
    else:
        start_time = str_datetime(splitted[0])
        end_time = np.nan
    return start_time, end_time


def my_preprocess_maiac_data(
    file_path: str,
    dataset_name: str="Optical_Depth_047",
    total_bounds: np.ndarray = None
):
    """
    Given a hdf file_path,
    create a GDF of each intersecting point and the accompanying
    dataset value (e.g. blue band AOD).
    Args:
        file_path (str): a path to a hdf
        grid_cell_gdf (gpd.GeoDataFrame): GeoDataFrame that contains,
            at a minimum, a `grid_id` and `geometry` column of Polygons.
        dataset_name (str): specific dataset name (e.g. "Optical_Depth_047").
        total_bounds (np.ndarray, optional): If provided, will filter out points that fall
            outside of these bounds. Composed of xmin, ymin, xmax, ymax.    
    Returns:
        GeoDataFrame that contains Points and associated values.
    """
    # Load blue band AOD data
    hdf = SD(file_path, SDC.READ) # Read into as a hdf file
    aod = hdf.select(dataset_name) # Select the desired dataset (Optical_Depth_47)
    shape = aod.info()[2] 

    # Calibrate and align data
    calibration_dict = aod.attributes() # 
    alignment_dict = create_alignment_dict(hdf) # Get the dict from hdf.attributes()...
    granule = get_granule(alignment_dict['upper_left'][0] )

    corrected_AOD = calibrate_data(aod, shape, calibration_dict)

    xv, yv = create_meshgrid(alignment_dict, shape) # Create a meshgrid from both corners stored in alignement dict & np.linspace()
    
    sinu_crs = Proj(f"+proj=sinu +R={alignment_dict['crs_params'][0]} +nadgrids=@null +wktext").crs
    lon, lat = transform_arrays(xv, yv, sinu_crs, wgs84_crs) # Transform meshgrid from sinusoidal to wgs84 using pyproj.Transformer

    # Save values that align with granules
    granule_gdf = convert_array_to_df(corrected_AOD, lat, lon, granule, wgs84_crs, total_bounds)

    print(f"N. pixeles: {len(granule_gdf)}")
        
    # create a geopandas df with columns( granule_id, aot value, polygon corners, aot value)
    #use lon & lat to return the values between grid_cell_gdf.total_bounds
    start_time, end_time = get_time_stamp(hdf)

    granule_gdf['start'] = start_time
    granule_gdf['end'] = end_time

    
    # Clean up files
    hdf.end()
    #return df
    return granule_gdf


from shapely.geometry import Polygon
import json


with open('./data/jsons/coords.json') as json_file:
    #polygons = json.load(json_file)
    coordinates = json.load(json_file)


def get_data_per_zone(zone: Tuple[str, List],
                      file_paths: List[str],
                      MAIAC_DIR: str,
                      EXPORT_DIR: str
                      ):

    """ Open each file and concatenate geoDF """

    for idx, file in enumerate(file_paths):
        
        file_path = MAIAC_DIR + file
        export_path = EXPORT_DIR + zone[0] + '.csv'
        gdf = my_preprocess_maiac_data(file_path, total_bounds=zone[1])

        if os.path.isfile(export_path):
            gdf.to_csv(export_path, mode='a', header=False)
        else: # else it exists so append without writing the header
            gdf.to_csv(export_path)
        
        print(f"Read file {idx}, ", sep=' ')



def get_data_all_zones(
    zones: Dict,
    file_paths: List[str],
    MAIAC_DIR: str="",
    EXPORT_DIR: str="",
    ):

    header = True
    for zone in zones.items():
        print(zone[0])
        get_data_per_zone(zone, file_paths, MAIAC_DIR, EXPORT_DIR)


