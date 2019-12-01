import geopandas as gpd
from pathlib import Path

class DataSet:
    def __init__(self, config):
        self._config=config
        self._train_config=self._config.config('data')
        self._train_data_base_path=self._train_config['TrainPath']
    
    def read_shape_file(self, area, year):
        area_year = area + '_' + year

        path = Path(self._train_data_base_path,area_year)
        # Read the first shapefile found
        shapefiles = list(path.glob('*.shp'))
        shapefile_abs_path = shapefiles[0].absolute()

        return gpd.read_file(shapefile_abs_path)
