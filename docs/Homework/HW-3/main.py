import os, keras
import numpy as np
import pandas as pd
from Model import buildModel
from Train import getTrain
from Predict import makePredict
from Base import getMaxConfidenceImage
from sklearn.metrics import confusion_matrix
from Plot import plotDigits, plotLossAccuracyCurves, plotModel, plotConfusionMatrix, plotSaliencyMap, plotImageFiltersResult, plotWhiteNoiseActivateFilters


def main(mode, DataGenerator):

    intVaildSize = 5000

    strProjectFolder = os.path.dirname(__file__)

    if DataGenerator:
        strOutputPath = "../02-Output/" + "gen" + mode
    else:
        strOutputPath = "../02-Output/" + mode

    DataTrain = np.load(os.path.join(strProjectFolder, "../data/Train.npz"))
    arrayLabel = DataTrain["Label.npy"]
    arrayTrainImage = DataTrain["Image.npy"]/255.
    arrayOneHotLabel = keras.utils.to_categorical(arrayLabel)

    arrayTrainX, arrayTrainY, arrayTrainLabel = arrayTrainImage[:-intVaildSize], arrayOneHotLabel[:-intVaildSize], arrayLabel[:-intVaildSize]
    arrayValidX, arrayValidY, arrayValidLabel = arrayTrainImage[-intVaildSize:], arrayOneHotLabel[-intVaildSize:], arrayLabel[-intVaildSize:]

    getTrain(arrayTrainX, arrayTrainY, arrayValidX, arrayValidY, DataGenerator, strProjectFolder, strOutputPath)
    
    plotLossAccuracyCurves(mode, strProjectFolder, strOutputPath)
    plotModel(mode, strProjectFolder, strOutputPath)
    
    listClasses = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    arrayPred = makePredict(mode, arrayX=arrayValidX, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
    arrayPredLabel = np.argmax(arrayPred, axis=1)
    arrayConfusionMatrix = confusion_matrix(arrayValidLabel, arrayPredLabel)
    plotConfusionMatrix(mode, arrayConfusionMatrix, listClasses=listClasses, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)

    dictMaxConfidence = getMaxConfidenceImage(mode, arrayX=arrayValidX, arrayYLabel=arrayValidLabel, listClasses=listClasses, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
    listShowId = dictMaxConfidence["IndexNP"]
    listShowImage = [arrayValidX[i] for i in listShowId] 
    listLabelX = ["True:" + dictMaxConfidence["Class"][i] +"\n"+ "Predict:" + dictMaxConfidence["Class"][j] for i, j in zip(dictMaxConfidence["LabelPP"], dictMaxConfidence["LabelNP"])]
    plotDigits(instances=listShowImage, intImagesPerRow=7, listLabelX=listLabelX, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)


    if mode == "cnn":
        plotSaliencyMap(mode, arrayX=arrayValidX, arrayYLabel=arrayValidLabel, listClasses=listClasses, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
        plotImageFiltersResult(mode, arrayX=arrayValidX, intChooseId=2, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
        plotWhiteNoiseActivateFilters(mode, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
    

if __name__ == "__main__":
    main(mode="cnn", DataGenerator=False)



