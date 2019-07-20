import os
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


def getTrain(arrayTrainX, arrayTrainY, arrayValidX, arrayValidY, DataGenerator, strProjectFolder, strOutputPath):

    intEpochs = 100
    intBatchSize = 128
    floatZoomRange = 0.2

    model = load_model(os.path.join(strProjectFolder, strOutputPath + "model.h5"))

    callbacks = []
    csvLogger = CSVLogger(os.path.join(strProjectFolder, strOutputPath + "log.csv"), separator=",", append=False)
    callbacks.append(csvLogger)

    if DataGenerator:
        genTrain = ImageDataGenerator(rotation_range=25,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1, # 水平或垂直投影變換
                                       zoom_range=[1-floatZoomRange, 1+floatZoomRange], # 按比例隨機縮放圖像尺寸
                                       horizontal_flip=True)
        genTrain.fit(arrayTrainX)
        model.fit_generator(genTrain.flow(arrayTrainX, arrayTrainY, batch_size=intBatchSize),
                            steps_per_epoch=3*arrayTrainX.shape[0]//intBatchSize, # 每次 batch 所要提取的sample數目為多少
                            verbose=2,
                            epochs=intEpochs, validation_data=(arrayValidX, arrayValidY), callbacks=callbacks, shuffle=True)
    else:
        model.fit(arrayTrainX, arrayTrainY, epochs=intEpochs, batch_size=intBatchSize, verbose=2, validation_data=(arrayValidX, arrayValidY), callbacks=callbacks, shuffle=True)

    model.save(os.path.join(strProjectFolder, strOutputPath + "model.h5"))