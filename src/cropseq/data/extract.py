import logging
import os

import fiona
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask

from cropseq import cfg

cfg.configLog()
"""
For each sample point, extracts land usage codes for each yearly raster. 
The script generates two files:
- resources/samples.pickle: panda dataframe with a row for each point in the shp file and two columns storing 
    the coordinates of the point in SRID 25830 
- resources/samples_sequence.pickle: panda dataframe with the points of samples.pickle and a column for each year 
    code extracted from the raster file. 
"""

def shp_to_df(shape_file):
    """
    Creates a dataframe from the shp file points data
    :return:
    """
    with fiona.open(shape_file, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    # get coordinales from shapely object
    coords = [p["coordinates"] for p in shapes]
    numpy_array = np.array(coords)
    df = pd.DataFrame(numpy_array)
    df.columns = ["x", "y"]
    return df


def load_coords_from_npy(input_file):
    logging.info("Loading coords from file " + input_file)
    return pd.read_pickle(input_file)


def add_LU_value(df, raster_file, col_name):
    logging.info("Adding land usage code from raster file {} into column {}".format(raster_file, col_name))

    with rasterio.open(raster_file, 'r') as src:
        values = list(src.sample(df[["x", "y"]].values))
    df[col_name] = values
    df[col_name] = df[col_name].astype(np.uint8)


def year_from_filepath(filepath):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    return base_name.split("_")[1]


def extract_LU_from_raster(df):
    """
    Recieves a panda dataframe with two columns, x,y that store the sample coordinates in 25830. For each point,
    the raster value of each year is extracted and store as column.
    :param df:
    :return: None
    """
    base_folder = cfg.resource("data/rasters")
    raster_files = [os.path.join(base_folder, x) for x in os.listdir(base_folder) if x.endswith(".tif")]
    raster_files.sort()

    # go over all raster files extracting then land_usage from the raster and storing in the dataframe
    for rfile in raster_files:
        # get year from file name
        year = year_from_filepath(rfile)
        add_LU_value(df, rfile, year)
        # temp backup
        df.to_pickle("/tmp/backup_{}.npy".format(year))


if __name__ == '__main__':
    # # shp to numpy coords file
    # coords_file = cfg.resource('samples.npy')
    # shape_file = cfg.resource('data/samples/samples.shp')
    # df = shp_to_df(shape_file)
    # df.to_pickle(coords_file)

    coords_file = cfg.resource('samples.pickle')
    df = load_coords_from_npy(coords_file)
    extract_LU_from_raster(df)

    dataset_file_name = cfg.resource("samples_sequence.pickle")
    df.to_pickle(dataset_file_name)
    logging.info("Dataset successfully created!!! {}".format(dataset_file_name))
    logging.info("Number of sample points: {}".format(df.shape[0]))
    logging.info("Number of Land usage columns: {}".format(df.shape[1] - 2))
    logging.info("First 10 values:")
    logging.info("\n{}".format(df.head()))
