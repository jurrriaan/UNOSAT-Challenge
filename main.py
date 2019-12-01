import app

def main():
    config = app.Config()
    dataset = app.DataSet(config)
    print(dataset.load_and_prepare_data('Mosul', '2015'))


if __name__ == "__main__":
    main()