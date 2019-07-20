import os
from keras.models import load_model


def makePredict(mode, arrayX, strProjectFolder, strOutputPath):

    strModelPath = os.path.join(strProjectFolder, strOutputPath + "model.h5")
    
    model = load_model(strModelPath)

    predictions = model.predict(arrayX)

    return predictions



