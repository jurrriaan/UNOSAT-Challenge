import geopandas as gpd
from pathlib import Path
import rasterio as rio
from rasterio import features
import numpy as np


def log(x, a, b):
    return np.log(a*x + b)

def normalize(array):
    """Normalize bands into 0.0 - 1.0 scale
    """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)
# https://stackoverflow.com/questions/11105375/how-to-split-a-matrix-into-4-blocks-using-numpy/51914911#51914911?newreg=399264720ba84b0cb7e27c4fb121322d
def split_array(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

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
        # Read raster_obj into np_array
        np_arr_read = raster_obj.read()
        np_arr = np_arr_read[0]

        # Read raster config
        raster_config = self._config.config('raster')

        # Clipping the array
        clip_min=float(raster_config['ClipMin'])
        clip_max=float(raster_config['ClipMax'])

        arr_clip = np.clip(np_arr, clip_min, clip_max)

        # Log array to spread distribution
        a_l = float(raster_config['ALog'])
        b_l = float(raster_config['BLog'])

        array_log = log(arr_clip, a_l, b_l)
        np_arr_log_norm = normalize(array_log)

        return np_arr_log_norm

    def transform_shape_file(self, shape_obj, raster_obj):
        # Transform to raster projection
        polygons_resh = shape_obj.to_crs(raster_obj.crs)
        # get polygon geometries
        geoms = polygons_resh['geometry']
        # Build polygon mask
        polygon_mask = features.geometry_mask(geometries=geoms,
                                       out_shape=(raster_obj.height, raster_obj.width),
                                       transform=raster_obj.transform,
                                       all_touched=False,
                                       invert=True)
        
        # Normalized polygon mask
        np_arr_mask = np.multiply(polygon_mask, 1)

        return np_arr_mask
    
    def crop_polygon_and_raster_arrays(self, np_arr_log_norm, np_arr_mask):
        np_arr_log_norm_crop = np_arr_log_norm[:-100,:-99]
        np_arr_mask_crop = np_arr_mask[:-100,:-99]


        return np_arr_log_norm_crop, np_arr_mask_crop
    
    def load_and_prepare_data(self, area, year):
        # Read shape and raster file
        shape_obj = self.read_shape_file(area, year)
        raster_obj = self.read_raster_file(area, year)
        # Transform shape and raster file
        np_arr_log_norm = self.transform_raster_file(raster_obj)
        np_arr_mask = self.transform_shape_file(shape_obj, raster_obj)
        # Crop files
        np_arr_log_norm_crop, np_arr_mask_crop = self.crop_polygon_and_raster_arrays(np_arr_log_norm, np_arr_mask)
        # Split into Xdata, Ydata
        size_sample = (128, 128)
        Xdata = split_array(np_arr_log_norm_crop, size_sample[0], size_sample[1])
        Ydata = split_array(np_arr_mask_crop, size_sample[0], size_sample[1])
        
        return Xdata, Ydata