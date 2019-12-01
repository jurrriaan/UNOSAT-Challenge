import app

def main():
    config = app.Config()
    dataset = app.DataSet(config)
    Xdata, Ydata = dataset.load_and_prepare_data('Mosul', '2015')
    i_show = 5000
    app.visualize_sample(Xdata[i_show], Ydata[i_show])


if __name__ == "__main__":
    main()