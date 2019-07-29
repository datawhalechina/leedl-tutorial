import csv, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
from Base import deprocessImage, makeNormalize, trainGradAscent


def plotDigits(instances, intImagesPerRow, listLabelX, strProjectFolder, strOutputPath):
    """
    This function display a lots of images together.
    """
    instances = np.array(instances).reshape(-1, 48, 48)
    intImagesPerRow = min(len(instances), intImagesPerRow)
    intNumRows = (len(instances) - 1) // intImagesPerRow + 1
    
    fig = plt.figure(figsize=(12, 2))
    for i in range(len(instances)):
        ax = fig.add_subplot(intNumRows, intImagesPerRow, i+1)
        ax.imshow(instances[i], cmap="gray")
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel("{}".format(listLabelX[i]))
        plt.tight_layout()
    plt.savefig(os.path.join(strProjectFolder, strOutputPath + "DisplayData"))


def plotModel(mode, strProjectFolder, strOutputPath):
    """
    This function plots the model structure.
    """
    model = load_model(os.path.join(strProjectFolder, strOutputPath + "model.h5"))
    model.summary()
    plot_model(model, show_shapes=True, to_file=os.path.join(strProjectFolder, strOutputPath + "model.png"))


def plotLossAccuracyCurves(mode, strProjectFolder, strOutputPath):
    """
    This function plots the Loss Curves and Accuracy Curves.
    """
    pdLog = pd.read_csv(os.path.join(strProjectFolder, strOutputPath + "log.csv"))

    fig = plt.figure(figsize=(12, 5))
    # Loss Curves
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(pdLog["epoch"], pdLog["loss"], "r", linewidth=1.5)
    plt.plot(pdLog["epoch"], pdLog["val_loss"], "b", linewidth=1.5)
    plt.legend(["Training loss", "Validation Loss"], fontsize=12)
    plt.xlabel("Epochs ", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title("Loss Curves", fontsize=10)
    # Accuracy Curves
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(pdLog["epoch"], pdLog["acc"], "r", linewidth=1.5)
    plt.plot(pdLog["epoch"], pdLog["val_acc"], "b", linewidth=1.5)
    plt.legend(["Training Accuracy", "Validation Accuracy"], fontsize=12)
    plt.xlabel("Epochs ", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.title("Accuracy Curves", fontsize=10)
    plt.savefig(os.path.join(strProjectFolder, strOutputPath + "LossAccuracyCurves"))


def plotConfusionMatrix(mode, confusionmatrix, listClasses, strProjectFolder, strOutputPath):
    """
    This function plots the confusion matrix.
    """  
    title = "ConfusionMatrix"
    cmap = plt.cm.jet

    confusionmatrix = confusionmatrix.astype("float") / confusionmatrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(confusionmatrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(listClasses))
    plt.xticks(tick_marks, listClasses, rotation=45)
    plt.yticks(tick_marks, listClasses)

    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        plt.text(j, i, "{:.2f}".format(confusionmatrix[i, j]), horizontalalignment="center",
                color="white" if confusionmatrix[i, j] > thresh else "black")
    plt.tight_layout() # 自動調整間距
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(strProjectFolder, strOutputPath+title))


def plotSaliencyMap(mode, arrayX, arrayYLabel, listClasses, strProjectFolder, strOutputPath):
    """
    This function plots the saliency map.
    """
    strModelPath = os.path.join(strProjectFolder, strOutputPath + "model.h5")

    model = load_model(strModelPath)

    inputImage = model.input

    listImageIDs = [23, 189, 68, 2, 6, 15, 4]
    for idx in listImageIDs:
        arrayProbability = model.predict(arrayX[idx].reshape(-1, 48, 48, 1))
        arrayPredictLabel = arrayProbability.argmax(axis=-1)
        tensorTarget = model.output[:, arrayPredictLabel] # ??
        tensorGradients = K.gradients(tensorTarget, inputImage)[0]
        fn = K.function([inputImage, K.learning_phase()], [tensorGradients])

        ### start heatmap processing ###
        arrayGradients = fn([arrayX[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
       
        arrayGradients = np.max(np.abs(arrayGradients), axis=-1, keepdims=True)

        # normalize center on 0., ensure std is 0.1
        arrayGradients = (arrayGradients - np.mean(arrayGradients)) / (np.std(arrayGradients) + 1e-5)
        arrayGradients *= 0.1

        # clip to [0, 1]
        arrayGradients += 0.5
        arrayGradients = np.clip(arrayGradients, 0, 1)

        arrayHeatMap = arrayGradients.reshape(48, 48)
        ### End heatmap processing ###
        
        print("ID: {}, Truth: {}, Prediction: {}".format(idx, arrayYLabel[idx], arrayPredictLabel))
        
        # show original image
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 3, 1)
        axx = ax.imshow((arrayX[idx]*255).reshape(48, 48), cmap="gray")
        plt.tight_layout()
        
        # show Heat Map
        ax = fig.add_subplot(1, 3, 2)
        axx = ax.imshow(arrayHeatMap, cmap=plt.cm.jet)
        plt.colorbar(axx)
        plt.tight_layout()

        # show Saliency Map
        floatThreshold = 0.55
        arraySee = (arrayX[idx]*255).reshape(48, 48)
        arraySee[np.where(arrayHeatMap <= floatThreshold)] = np.mean(arraySee)

        ax = fig.add_subplot(1, 3, 3)
        axx = ax.imshow(arraySee, cmap="gray")
        plt.colorbar(axx)
        plt.tight_layout()
        fig.suptitle("Class {}".format(listClasses[listImageIDs.index(idx)]))
        plt.savefig(os.path.join(strProjectFolder, strOutputPath + "SaliencyMap" + listClasses[listImageIDs.index(idx)]))


def plotWhiteNoiseActivateFilters(mode, strProjectFolder, strOutputPath):
    """
    This function plot Activate Filters with white noise as input images
    """
    intRecordFrequent = 20
    intNumberSteps = 160
    intIterationSteps = 160

    strModelPath = os.path.join(strProjectFolder, strOutputPath + "model.h5")

    model = load_model(strModelPath)
    dictLayer = dict([layer.name, layer] for layer in model.layers)
    inputImage = model.input
    listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
    listCollectLayers = [dictLayer[name].output for name in listLayerNames]

    for cnt, fn in enumerate(listCollectLayers):
        listFilterImages = []
        intFilters = 64
        for i in range(intFilters):
            arrayInputImage = np.random.random((1, 48, 48, 1)) # random noise
            tensorTarget = K.mean(fn[:, :, :, i])

            tensorGradients = makeNormalize(K.gradients(tensorTarget, inputImage)[0])
            targetFunction = K.function([inputImage, K.learning_phase()], [tensorTarget, tensorGradients])

            # activate filters
            listFilterImages.append(trainGradAscent(intIterationSteps, arrayInputImage, targetFunction, intRecordFrequent))
        
        for it in range(intNumberSteps//intRecordFrequent):
            print("In the #{}".format(it))
            fig = plt.figure(figsize=(16, 17))
            for i in range(intFilters):
                ax = fig.add_subplot(intFilters/8, 8, i+1)
                arrayRawImage = listFilterImages[i][it][0].squeeze()
                ax.imshow(deprocessImage(arrayRawImage), cmap="Blues")
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel("{:.3f}".format(listFilterImages[i][it][1]))
                plt.tight_layout()
            fig.suptitle("Filters of layer {} (# Ascent Epoch {} )".format(listLayerNames[cnt], it*intRecordFrequent))
            plt.savefig(os.path.join(strProjectFolder, strOutputPath + "FiltersWhiteNoise" + listLayerNames[cnt]))


def plotImageFiltersResult(mode, arrayX, intChooseId, strProjectFolder, strOutputPath):
    """
    This function plot the output of convolution layer in valid data image.
    """
    intImageHeight = 48
    intImageWidth = 48

    strModelPath = os.path.join(strProjectFolder, strOutputPath + "model.h5")

    model = load_model(strModelPath)
    dictLayer = dict([layer.name, layer] for layer in model.layers)
    inputImage = model.input
    listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
    # define the function that input is an image and calculate the image through each layer until the output layer that we choose
    listCollectLayers = [K.function([inputImage, K.learning_phase()], [dictLayer[name].output]) for name in listLayerNames] 

    for cnt, fn in enumerate(listCollectLayers):
        arrayPhoto = arrayX[intChooseId].reshape(1, intImageWidth, intImageHeight, 1)
        listLayerImage = fn([arrayPhoto, 0]) # get the output of that layer list (1, 1, 48, 48, 64)
        
        fig = plt.figure(figsize=(16, 17))
        intFilters = 64
        for i in range(intFilters):
            ax = fig.add_subplot(intFilters/8, 8, i+1)
            ax.imshow(listLayerImage[0][0, :, :, i], cmap="Blues")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel("filter {}".format(i))
            plt.tight_layout()
        fig.suptitle("Output of {} (Given image{})".format(listLayerNames[cnt], intChooseId))
        plt.savefig(os.path.join(strProjectFolder, strOutputPath + "FiltersResultImage" + str(intChooseId) + listLayerNames[cnt]))

