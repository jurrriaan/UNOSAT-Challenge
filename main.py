import app

def main():
    config = app.Config()
    dataset = app.DataSet(config)
    print(dataset.read_shape_file('Mosul', '2015'))

if __name__ == "__main__":
    main()