import geopandas as gpd
from pathlib import Path
import rasterio as rio

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