import geopandas as gpd
from pathlib import Path
import rasterio as rio
import numpy as np

def log(x, a, b):
    return np.log(a*x + b)

def normalize(array):
    """Normalize bands into 0.0 - 1.0 scale
    """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

class DataSet:
    def __init__(self, config):
        self._config=config
        self._train_config=self._config.config('data')
        self._train_data_base_path=self._train_config['TrainPath']
    
    def read_shape_file(self, area, year):
        """Read shapefile from disk into geopandas object

        """
        area_year = area + '_' + year

        path = Path(self._train_data_base_path,area_year)
        # Read the first shapefile found
        shape_files = list(path.glob('*.shp'))
        shape_file_abs_path = shape_files[0].absolute()

        return gpd.read_file(shape_file_abs_path)

    def read_raster_file(self, area, year):
        """Read raster file from disk

        """
        # TODO find a way to search for raster files dynamically
        area_year = area + '_' + year
        raster_file_path = self._train_config['RasterFilePath']
        path = Path(self._train_data_base_path, area_year, raster_file_path)
        raster_file_abs_path = path.absolute()
        
        return rio.open(raster_file_abs_path)
    
    def transform_raster_file(self, raster_obj):
        """Take a rio opened dataset and read it into a np array, performing some transformations.

        """
        np_arr_read = raster_obj.read()
        np_arr = np_arr_read[0]

        # Clipping the array
        clip_min=0
        clip_max=2

        arr_clip = np.clip(np_arr, clip_min, clip_max)

        # Log array to spread distribution
        a_l = 1
        b_l = 0.1

        array_log = log(arr_clip, a_l, b_l)
        np_arr_log_norm = normalize(array_log)

        return np_arr_log_norm