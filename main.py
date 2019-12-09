import src

def main():
    config = src.Config()
    dataset = src.DataSet(config)
    x_data, y_data = dataset.load_and_prepare_data('Mosul', '2015')
    i_show = 5000
    src.visualize_sample(x_data[i_show], y_data[i_show])
    ml = src.ML(config, x_data, y_data)
    ml.train()


if __name__ == "__main__":
    main()