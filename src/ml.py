from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras_unet.models import satellite_unet 
from keras_unet.metrics import iou, iou_thresholded

class ML:
    def __init__(self, config, x_data, y_data):
        self.config = config
        self.ml_config = self.config.config("ml")
        x_data_input = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
        y_data_input = y_data.reshape(y_data.shape[0], y_data.shape[1], y_data.shape[2], 1)
        
        test_size=float(self.ml_config["TestSize"])
        (train_x, test_x, train_y, test_y) = train_test_split(x_data_input,  y_data_input, test_size=0.2, random_state=42)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def train(self):
        image_dims=(128, 128, 1)
        model = satellite_unet(image_dims)
        init_lr = float(self.ml_config["InitLr"])
        epochs = int(self.ml_config["Epochs"])
        opt = Adam(lr=init_lr, decay=init_lr / epochs)
        model.compile(loss="binary_crossentropy", 
              optimizer=opt, 
              metrics=[iou, iou_thresholded])
        
        bs = int(self.ml_config["Bs"])
        H = model.fit(self.train_x, self.train_y,  
                 batch_size=bs, 
                epochs=epochs, 
                verbose=1)
                            
        # evaluate the model
        scores = model.evaluate(self.test_x, self.test_y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))