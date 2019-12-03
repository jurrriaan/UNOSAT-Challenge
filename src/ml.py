from sklearn.model_selection import train_test_split

class ML:
    def __init__(self, config, x_data, y_data):
        self.config = config
        self.ml_config = self.config.config("ml")
        x_data_input = x_data.reshape(x_data.shape[0], x_dataata.shape[1], x_data.shape[2], 1)
        y_data_input = y_data.reshape(y_data.shape[0], y_data.shape[1], y_data.shape[2], 1)

        (train_x, test_x, train_y, test_y) = train_test_split(x_data_input,  y_data_input, test_size=self.ml_config["TestSize"], random_state=42)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y