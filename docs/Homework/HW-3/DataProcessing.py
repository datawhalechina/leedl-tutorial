import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Plot import plotDigits


def makeDataProcessing(Data):
    listLabel = []
    listImageVector = []
    listImage = []
    for index, strRow in enumerate(Data):
        strLabel, strImageVector = strRow.split(",")
        if index != 0:
            arrayLabel = int(strLabel)
            arrayImageVector = np.fromstring(strImageVector, dtype=int, sep=" ") # for dnn
            arrayImage = arrayImageVector.reshape(48, 48, 1) # for cnn

            listLabel.append(arrayLabel)
            listImageVector.append(arrayImageVector)
            listImage.append(arrayImage)
    return listLabel, listImageVector, listImage


if __name__ == "__main__":

    strProjectFolder = os.path.dirname(__file__)
    strOutputPath = "02-Output/"

    DataTrain = open(os.path.join(strProjectFolder, "./data/train.csv"), "r")
    DataTest = open(os.path.join(strProjectFolder, "./data/test.csv"), "r")

    listTrainLabel, listTrainImageVector, listTrainImage = makeDataProcessing(DataTrain)
    np.savez(os.path.join(strProjectFolder, "./data/Train.npz"), Label=np.asarray(listTrainLabel), Image=np.asarray(listTrainImage))

    _, listTestImageVector, listTestImage = makeDataProcessing(DataTest)
    np.savez(os.path.join(strProjectFolder, "./data/Test.npz"), Image=np.asarray(listTestImage))

    listShowId = [0, 299, 2, 7, 3, 15, 4]
    listShowImage = [listTrainImage[i] for i in listShowId] 
    listLabelX = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    plotDigits(instances=listShowImage, intImagesPerRow=7, listLabelX=listLabelX, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)



