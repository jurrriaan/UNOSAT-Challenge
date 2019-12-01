import app

def main():
    config = app.Config()
    dataset = app.DataSet(config)
    print(dataset.read_shape_file('Mosul', '2015'))
    raster_obj = dataset.read_raster_file('Mosul', '2015')
    print(dataset.transform_raster_file(raster_obj))


if __name__ == "__main__":
    main()