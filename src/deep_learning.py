
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.utils import get_augmented
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np


def reshape_data(trainX, trainY):
    print('reshape data...')
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[2], trainX.shape[3], trainX.shape[1])
    trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], trainY.shape[2], 1)
    return trainX, trainY


def init_and_fit_model(trainX, trainY, model, BS, EPOCHS, opt):
    print('augment data...')
    train_datagen = get_augmented(
        trainX,
        trainY,
        X_val=None,
        Y_val=None,
        batch_size=BS,
        seed=0,
        data_gen_args=dict(
            rotation_range=90,
            height_shift_range=0,
            shear_range=0,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant'
        )
    )

    print('compile model...')
    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  # metrics=["accuracy"],
                  metrics=[iou, iou_thresholded])


    filepath ="weights-improvement-{epoch:02d}-{iou:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='iou', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print('start fit...')
    H = model.fit_generator(generator=train_datagen,
                            steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS,
                            callbacks=callbacks_list,
                            verbose=2
                            )

    print('save model and weights')
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_epochs.h5")

    return model, H


def plot_info(H, EPOCHS):
    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["iou"], label="train_iou")

    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/iou")
    plt.legend(loc="lower left")
    plt.savefig("plot")

