from keras import backend as K
import numpy as np
from keras.models import load_model
import os


def getMaxConfidenceImage(mode, arrayX, arrayYLabel, listClasses, strProjectFolder, strOutputPath):
    """
    For each class, This function finds the max confidence prediction label for both the prediction label is right and wrong.
    """
    model = load_model(os.path.join(strProjectFolder, strOutputPath + "model.h5"))

    listIndexPP = []
    listIndexNP = []
    listLabelPP = []
    listLabelNP = []
    listConfidencePP = [] 
    listConfidenceNP = []
    for c in range(len(listClasses)):
        arrayProbPP = np.asarray([0.0])
        arrayProbNP = np.asarray([0.0])
        for idx in np.where(arrayYLabel == c)[0]:
            arrayProbability = model.predict(arrayX[idx].reshape(-1, 48, 48, 1))
            arrayPredictLabel = arrayProbability.argmax(axis=-1)
            arrayConfidence = arrayProbability[0][arrayPredictLabel]

            if arrayConfidence > arrayProbPP and arrayPredictLabel[0] == c:
                arrayProbPP = arrayConfidence
                arrayIndexPP = np.array([idx])
                arrayLabelPP = arrayPredictLabel

            if arrayConfidence > arrayProbNP and arrayPredictLabel[0] != c:
                arrayProbNP = arrayConfidence
                arrayIndexNP = np.array([idx])
                arrayLabelNP = arrayPredictLabel

        listIndexPP.extend(arrayIndexPP)
        listLabelPP.extend(arrayLabelPP)
        listConfidencePP.extend(arrayProbPP)

        listIndexNP.extend(arrayIndexNP)
        listLabelNP.extend(arrayLabelNP)
        listConfidenceNP.extend(arrayProbNP)

    dictSummaryTable = {}
    dictSummaryTable["Class"] = listClasses
    dictSummaryTable["IndexPP"] = listIndexPP
    dictSummaryTable["LabelPP"] = listLabelPP
    dictSummaryTable["ConfidencePP"] = listConfidencePP
    dictSummaryTable["IndexNP"] = listIndexNP
    dictSummaryTable["LabelNP"] = listLabelNP
    dictSummaryTable["ConfidenceNP"] = listConfidenceNP

    return dictSummaryTable


# util function to convert a tensor into a valid image
def deprocessImage(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # print(x.shape)
    return x


def makeNormalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)


def trainGradAscent(intIterationSteps, arrayInputImageData, targetFunction, intRecordFrequent):
    """
    Implement gradient ascent in targetFunction
    """
    listFilterImages = []
    floatLearningRate = 1e-2
    for i in range(intIterationSteps):
        floatLossValue, arrayGradientsValue = targetFunction([arrayInputImageData, 0])
        arrayInputImageData += arrayGradientsValue * floatLearningRate
        if i % intRecordFrequent == 0:
            listFilterImages.append((arrayInputImageData, floatLossValue))
            print("#{}, loss rate: {}".format(i, floatLossValue))
    return listFilterImages
