import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import csv
from scipy.stats import pearsonr
import random
import math
import os
import copy
import pydot

seed = 1234
def getNextSeed():
    global seed
    next_seed = seed
    seed = seed + 1
    return next_seed

def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

# Order_x: donki_speed,	donki_ha, longitude, latitude, Accel, Type_2_Area, 2nd_order_speed_final, 2nd_order_speed_20R, Central_PA, MPA, sunspots, CMEs_past_month, CMEs_past_9_hours, CMEs_over_1000_past_9_hrs, Max_speed_past_day, richardson_formula_degrees_phi_2_solar_wind, V log V, halo, diffusive_shock, Double_CME_100_MeV
# Order_y: 100MeV_peak_intensity, threshold_time, peak_time, 100MeV_peak_intensity_ln
def getFirstStageInput(csvFile):
    data = readCSVFile(csvFile)
    
    numElements = len(data)
    inputData_x = np.empty((numElements, 20), dtype=np.float64)
    inputData_y = np.empty((numElements, 1), dtype=np.float64)
    
    for i in range(0, numElements):
        elem = data[i]
        
        inputData_x[i][0] = float(elem["donki_speed"])
        inputData_x[i][1] = float(elem["donki_ha"])
        inputData_x[i][2] = float(elem["longitude"])
        inputData_x[i][3] = float(elem["latitude"])
        inputData_x[i][4] = float(elem["Accel"])
        inputData_x[i][5] = float(elem["Type_2_Area"])
        inputData_x[i][6] = float(elem["2nd_order_speed_final"])
        inputData_x[i][7] = float(elem["2nd_order_speed_20R"])
        inputData_x[i][8] = float(elem["Central_PA"])
        inputData_x[i][9] = float(elem["MPA"])
        inputData_x[i][10] = float(elem["sunspots"])
        inputData_x[i][11] = float(elem["CMEs_past_month"])
        inputData_x[i][12] = float(elem["CMEs_past_9_hours"])
        inputData_x[i][13] = float(elem["CMEs_over_1000_past_9_hrs"])
        inputData_x[i][14] = float(elem["Max_speed_past_day"])
        inputData_x[i][15] = float(elem["richardson_formula_degrees_phi_2_solar_wind"])
        inputData_x[i][16] = float(elem["V log V"])
        inputData_x[i][17] = float(elem["halo"])
        inputData_x[i][18] = float(elem["diffusive_shock"])
        inputData_x[i][19] = float(elem["Double_CME_100_MeV"])
        
        inputData_y[i][0] = float(elem["100MeV_peak_intensity_ln"])
        if inputData_y[i][0] > 0:
            inputData_y[i][0] = 1
        else:
            inputData_y[i][0] = 0
        #inputData_y[i][1] = float(elem["threshold_time"])
        #inputData_y[i][2] = float(elem["peak_time"])
    
    return (inputData_x, inputData_y)

def outputPredictions(data_x, data_y, pred_y, target_y, csvFilename, pred_threshold):
    with open(csvFilename, 'w', newline='') as csvfile:
        fieldnames = ["dummy", "index", "donki_speed", "donki_ha", "longitude", "latitude", "Accel", "Type_2_Area", "2nd_order_speed_final", "2nd_order_speed_20R", "Central_PA", "MPA", "sunspots", "CMEs_past_month", "CMEs_past_9_hours", "CMEs_over_1000_past_9_hrs", "Max_speed_past_day", "richardson_formula_degrees_phi_2_solar_wind", "V log V", "halo", "diffusive_shock", "Double_CME_100_MeV", "target", "100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln", "predicted_thresholded"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()        
        
        for i in range(0, len(data_x)):
            thresholded_value = 1
            if pred_y[i][0] < pred_threshold:
                thresholded_value = 0
            row = {
                "dummy" : 0,
                "index" : (i + 1),
                "donki_speed" : data_x[i]["donki_speed"],
                "donki_ha" : data_x[i]["donki_ha"],
                "longitude" : data_x[i]["longitude"],
                "latitude" : data_x[i]["latitude"],
                "Accel" : data_x[i]["Accel"],
                "Type_2_Area" : data_x[i]["Type_2_Area"],
                "2nd_order_speed_final" : data_x[i]["2nd_order_speed_final"],
                "2nd_order_speed_20R" : data_x[i]["2nd_order_speed_20R"],
                "Central_PA" : data_x[i]["Central_PA"],
                "MPA" : data_x[i]["MPA"],
                "sunspots" : data_x[i]["sunspots"],
                "CMEs_past_month" : data_x[i]["CMEs_past_month"],
                "CMEs_past_9_hours" : data_x[i]["CMEs_past_9_hours"],
                "CMEs_over_1000_past_9_hrs" : data_x[i]["CMEs_over_1000_past_9_hrs"],
                "Max_speed_past_day" : data_x[i]["Max_speed_past_day"],
                "richardson_formula_degrees_phi_2_solar_wind" : data_x[i]["richardson_formula_degrees_phi_2_solar_wind"],
                "V log V" : data_x[i]["V log V"],
                "halo" : data_x[i]["halo"],
                "diffusive_shock" : data_x[i]["diffusive_shock"],
                "Double_CME_100_MeV" : data_x[i]["Double_CME_100_MeV"],
                "target" : target_y[i],
                "100MeV_peak_intensity_ln" : data_y[i][0],
                "predicted_100MeV_peak_intensity_ln" : pred_y[i][0],
                "predicted_thresholded" : thresholded_value
                }
            writer.writerow(row)
    return

def getTargets(csvFile):
    data = readCSVFile(csvFile)
    numElements = len(data)

    keysData = np.empty((numElements), dtype=np.intc)
    
    for i in range(0, numElements):
        elem = data[i]
        
        keysData[i] = int(elem["target"])
    
    return keysData

def getValidationData(csvFile):
    # Validation data should be tuples (x_val, y_val) of Numpy arrays or tensors
    data_x, data_y = getFirstStageInput(csvFile)

    validationData = (data_x, data_y)
    
    return validationData

def showHistoryValidation(history, show=False):
    if history is None:
        print("History is None...")
        return
    lossHistory = history.history["loss"]
    valLossHistory = history.history["val_loss"]
    xAxisValues = []
    yAxisValuesLoss = []
    yAxisValuesValLoss = []
    minValLossIndex = 0
    minLossIndex = 0
    for i in range(0, len(lossHistory)):
        epoch = i + 1
        xAxisValues.append(epoch)
        yAxisValuesLoss.append(lossHistory[i])
        yAxisValuesValLoss.append(valLossHistory[i])
        if valLossHistory[i] < valLossHistory[minValLossIndex]:
            minValLossIndex = i
        if lossHistory[i] < lossHistory[minLossIndex]:
            minLossIndex = i

    if show:
        plt.title("Loss (Cross Entropy) and Validation Loss VS Epoch")
        plt.xlabel("Epoch")
        plt.scatter(xAxisValues, yAxisValuesLoss, label="Loss (Cross Entroy)")
        plt.scatter(xAxisValues, yAxisValuesValLoss, label="Validation Loss")
        plt.legend()
        plt.show()
        print("Min loss index: {:}, Min loss: {:.5f}".format(minLossIndex, lossHistory[minLossIndex]))
        print("Min val loss index: {:}, Min val loss: {:.5f}".format(minValLossIndex, valLossHistory[minValLossIndex]))

def showHistory(history, historyKey="loss", plotTitle="Loss (Cross Entropy) VS Epoch", plotYLabel="Loss (Cross Entropy)", show=False):
    if history is None:
        print("History is None...")
        return
    keyHistory = history.history[historyKey]
    xAxisValues = []
    yAxisValues = []
    for i in range(0, len(keyHistory)):
        epoch = i + 1
        xAxisValues.append(epoch)
        yAxisValues.append(keyHistory[i])

    if show:
        plt.title(plotTitle)
        plt.xlabel("Epoch")
        plt.ylabel(plotYLabel)
        plt.scatter(xAxisValues, yAxisValues)
        plt.show()

def LeakyReLU(alpha=0.2):
    return lambda x: tf.keras.backend.maximum(alpha * x, x)

def regNNModelValidationWithAdamAndLeakyReLU(adamLearningRate, trainingData_x, trainingData_y, validationData, logFilename, adamEpsilon, alpha, numEpochs=3000, callbacks=[], weights_filename=None):
    print("Adam Learning Rate: {:.5f}. Adam epsilon: {:.5f}. Num epochs: {:}. Alpha: {:.5f}.".format(adamLearningRate, adamEpsilon, numEpochs, alpha))
        
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())
    # Input layer
    inputLayer = keras.Input(shape=(20,))

    # Hidden layer
    encoderLayer = keras.layers.Dense(18, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(inputLayer)

    zLayer = keras.layers.Dense(9, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(encoderLayer)

    # Output Layer
    outputLayer = layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(zLayer)

    model = keras.Model(inputs=inputLayer, outputs=outputLayer, name="adapted_rRT_" + str(adamLearningRate))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.BinaryCrossentropy())

    if weights_filename is not None:
        model.load_weights(weights_filename)
        history = None

    csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
    callbacks.append(csv_logger)
    history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, validation_data=validationData, callbacks=callbacks, verbose=0)
    showHistoryValidation(history)
    
    return (model, history)

def regNNModelWithAdamAndLeakyReLU(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, seed, adamEpsilon, alpha, numEpochs=150, callbacks=[], weights_filename=None):
    print("Adam Learning Rate: {:.5f}. Adam epsilon: {:.5f}. Num epochs: {:}. Alpha: {:.5f}.".format(adamLearningRate, adamEpsilon, numEpochs, alpha))
        
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
    # Input layer
    inputLayer = keras.Input(shape=(20,))

    # Hidden layer
    encoderLayer = keras.layers.Dense(18, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(inputLayer)

    zLayer = keras.layers.Dense(9, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(encoderLayer)

    # Output Layer
    outputLayer = layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(zLayer)

    model = keras.Model(inputs=inputLayer, outputs=outputLayer, name="adapted_rRT_" + str(adamLearningRate))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.BinaryCrossentropy())
    
    if weights_filename is not None:
        model.load_weights(weights_filename)
        history = None
    else:
        csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
        callbacks.append(csv_logger)
        history = model.fit(all_trainingData_x, all_trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0)
        showHistory(history)
    
    return (model, history)

def discretize(predictedData_y, threshold=0.5):
    discretized_predicted_y = []
    for y in predictedData_y:
        if y[0] > threshold:
            discretized_predicted_y.append(1)
        else:
            discretized_predicted_y.append(0)
    return discretized_predicted_y

def plotEvaluationLN(y_true, y_predictions, y_target, show=False):
    predictions = discretize(y_predictions)

    minX = min(y_true) - 0.1
    maxX = max(y_true) + 0.1
    minY = min(predictions) - 0.1
    maxY = max([max(predictions) + 0.1, 1.1])

    y_true_SEP = []
    predictions_SEP = []

    scatterColors = []
    for i in range(0, len(y_target)):
        target = y_target[i]
        color = "black"
        if target == 0:
            # Background
            color = "blue"
        elif target == 1:
            # SEP
            color = "red"

            y_true_SEP.append(y_true[i])
            predictions_SEP.append(predictions[i])
        elif target == 2:
            # Elevated
            color = "green"
        scatterColors.append(color)

    corr, _ = pearsonr(np.reshape(y_true, len(y_true)), np.reshape(predictions, len(predictions)))
    
    # Is CME related to SEP Event | Actual No | Actual Yes
    # Predicted No                | TN        | FN
    # Predicted Yes               | FP        | TP
    TN = 0
    FN = 0
    FP = 0
    TP = 0
    for i in range(0, len(y_true)):
        actualY = y_true[i]
        predictedY = predictions[i]
        if actualY == 1:
            # Actual Yes
            if predictedY == 1:
                # Predicted Yes
                TP = TP + 1
            else:
                # Predictred No
                FN = FN + 1
        else:
            # Actual No
            if predictedY == 1:
                # Predicted Yes
                FP = FP + 1
            else:
                # Predicted No
                TN = TN + 1

    precisionDenominator = TP + FP
    if math.isclose(precisionDenominator, 0.0):
        precision = 0
    else:
        precision = TP / precisionDenominator

    # recall = TPR = TP / (TP + FN)
    recallDenominator = TP + FN
    if math.isclose(recallDenominator, 0.0):
        recall = 0
    else:
        recall = TP / recallDenominator
    TPR = recall
    
    F1Denominator = precision + recall
    if math.isclose(F1Denominator, 0.0):
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / F1Denominator
    
    FPRDenominator = FP + TN
    if math.isclose(F1Denominator, 0.0):
        FPR = 0
    else:
        FPR = FP / FPRDenominator

    TSS = TPR - FPR
    
    HSSDenominator = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
    if math.isclose(HSSDenominator, 0.0):
        HSS = 0
    else:
        HSS = 2 * (TP * TN - FP * FN) / HSSDenominator

    # Plot y-axis as predicted value and x-axis as original value
    plotTitle = "Predicted Value vs Original Value"

    if show:
        fig, ax = plt.subplots()

        ax.scatter(y_true, predictions, c=scatterColors)
        ax.set_title(plotTitle)
        ax.set_xlabel("Original Value")
        ax.set_ylabel("Predicted Value")
        ax.set_xlim(minX, maxX)
        ax.set_ylim(minY, maxY)

        dottedDiagLine = mlines.Line2D([0, 1], [0, 1], color='black', ls="--")
        thresholdVerticalLine = mlines.Line2D([1, 1], [minY, maxY], color='black', ls="--")
        thresholdHorizontalLine = mlines.Line2D([minX, maxX], [1, 1], color='black', ls="--")

        transform = ax.transAxes
        dottedDiagLine.set_transform(transform)

        ax.add_line(dottedDiagLine)
        ax.add_line(thresholdVerticalLine)
        ax.add_line(thresholdHorizontalLine)

        ax.text(4, 0.5, "Pearson: {:.2f}".format(corr))
        ax.text(4, 5.5, "F1: {:.2f}".format(F1))
        ax.text(4, 8.5, "HSS: {:.2f}".format(HSS))
        ax.text(4, 11.5, "TSS: {:.2f}".format(TSS))

        plt.show()
    
    return (corr, FP, FN, TP, TN, F1, HSS, TSS)

# Second stage use class based sampling to retrain regressive weights. 
def rRTValidationAndLeakyReLU(adamLearningRate, model, trainingData_x, trainingData_y, validationData, logFilename, adamEpsilon, alpha, numEpochs=3000, callbacks=[], weights_filename=None, do_train=True):
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

    # Freeze all but the last layer

    # Input layer
    inputLayerRetrain = keras.Input(shape=(20,))

    # Hidden layer
    encoderLayerWeights = model.layers[1].get_weights()
    encoderLayerRetrain = keras.layers.Dense(18, activation=LeakyReLU(alpha=alpha), trainable=False, weights=encoderLayerWeights)(inputLayerRetrain)

    zLayerWeights = model.layers[2].get_weights()
    zLayerRetrain = keras.layers.Dense(9, activation=LeakyReLU(alpha=alpha), trainable=False, weights=zLayerWeights)(encoderLayerRetrain)

    nonLinearLayer = keras.layers.Dense(6, activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer)(zLayerRetrain)

    # Reinitialize classifier weights
    outputLayerRetrain = layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(nonLinearLayer)

    modelRetrain = keras.Model(inputs=inputLayerRetrain, outputs=outputLayerRetrain, name="adapted_rRT_retrain")
    modelRetrain.summary()

    # Optimizer = model.optimizer
    # Loss = Mean Squared Error
    modelRetrain.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.BinaryCrossentropy())
    
    if weights_filename is not None:
        historyRetrain = None
        modelRetrain.load_weights(weights_filename)
    if do_train:
        csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
        callbacks.append(csv_logger)
    
        historyRetrain = modelRetrain.fit(trainingData_x, trainingData_y, epochs=numEpochs, validation_data=validationData, callbacks=callbacks, verbose=0)
        showHistoryValidation(historyRetrain)
    
    return (modelRetrain, historyRetrain)

# Second stage use class based sampling to retrain regressive weights. 
def rRTAndLeakyReLU(adamLearningRate, model, all_trainingData_x, all_trainingData_y, logFilename, seed, adamEpsilon, alpha, numEpochs=3000, weights_filename=None):
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

    # Freeze the all but the last layer

    # Input layer
    inputLayerRetrain = keras.Input(shape=(20,))

    # Hidden layer
    encoderLayerWeights = model.layers[1].get_weights()
    encoderLayerRetrain = keras.layers.Dense(18, activation=LeakyReLU(alpha=alpha), trainable=False, weights=encoderLayerWeights)(inputLayerRetrain)

    zLayerWeights = model.layers[2].get_weights()
    zLayerRetrain = keras.layers.Dense(9, activation=LeakyReLU(alpha=alpha), trainable=False, weights=zLayerWeights)(encoderLayerRetrain)

    nonLinearLayer = keras.layers.Dense(6, activation=LeakyReLU(alpha=alpha), kernel_initializer=initializer)(zLayerRetrain)

    # Reinitialize classifier weights
    outputLayerRetrain = layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(nonLinearLayer)

    modelRetrain = keras.Model(inputs=inputLayerRetrain, outputs=outputLayerRetrain, name="adapted_rRT_all_retrain")
    modelRetrain.summary()

    # Optimizer = model.optimizer
    # Loss = Mean Squared Error
    modelRetrain.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.BinaryCrossentropy())
    
    if weights_filename is not None:
        historyRetrain = None
        modelRetrain.load_weights(weights_filename)
    else:
        csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
        callbacks = []
        callbacks.append(csv_logger)
        
        historyRetrain = modelRetrain.fit(all_trainingData_x, all_trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0)
        #showHistory(historyRetrain)
    
    return (modelRetrain, historyRetrain)

def rRTAndLeakyReLURecall(adamLearningRate, model, weights_filename, adamEpsilon, alphaf):
    modelRetrain, _ = rRTAndLeakyReLU(adamLearningRate, model, None, None, None, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)
    
    modelRetrain.load_weights(weights_filename)
    
    return modelRetrain

def autoencoder_finding_alpha(optimizer, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilename, logFilenameClassifier, numEpochs, weights_filename, weights_filename_classifier, alpha):
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

    # Autoencoder
    # Input layer
    inputLayerFirst = keras.layers.Input(shape=(20,))

    # Encoder layer
    encoderLayerFirst = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='encoder'
    )(inputLayerFirst)

    # z-layer
    zLayerFirst = keras.layers.Dense(
        9,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='z-layer'
    )(encoderLayerFirst)

    decoderLayerFirst = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='decoder'
    )(zLayerFirst)

    autoencoderLayerFirst = keras.layers.Dense(
        20,
        activation=None,
        kernel_initializer=initializer, 
        name='autoencoder'
    )(decoderLayerFirst)

    outputFirst = autoencoderLayerFirst

    modelFirst = keras.Model(inputs=inputLayerFirst, outputs=outputFirst, name="autoencoder_model")
    modelFirst.summary()

    modelFirst.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    callbacks = []
    csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
    callbacks.append(csv_logger)
    historyFirst = modelFirst.fit(autoencoder_trainingData_x, autoencoder_trainingData_x, epochs=numEpochs, callbacks=callbacks, validation_data=(autoencoder_validationData[0], autoencoder_validationData[0]), verbose=0)

    modelFirst.save_weights(weights_filename)

    # Classifier
    # Input layer
    inputLayerSecond = keras.layers.Input(shape=(20,))
    
    # Encoder layer
    encoderLayerSecond = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='encoder'
    )(inputLayerSecond)
    
    # z-layer
    zLayerSecond = keras.layers.Dense(
        9,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='z-layer'
    )(encoderLayerSecond)
    
    classiferLayerSecond = keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=initializer, 
        name='classifier'
    )(zLayerSecond)

    outputSecond = classiferLayerSecond

   
    modelSecond = keras.Model(inputs=inputLayerSecond, outputs=outputSecond, name="autoencoder_classifier_model")
    modelSecond.summary()

    modelSecond.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy())

    callbacksClassifier = []
    csv_logger_classifier = keras.callbacks.CSVLogger(logFilenameClassifier, append=True, separator=';')
    callbacksClassifier.append(csv_logger_classifier)
    historySecond = modelSecond.fit(autoencoder_trainingData_x, autoencoder_trainingData_y, epochs=numEpochs, callbacks=callbacksClassifier, validation_data=autoencoder_validationData, verbose=0)

    modelSecond.save_weights(weights_filename_classifier)

    return (modelFirst, modelSecond)

def autoencoder(optimizer, lossWeightAlpha, logFilename, trainingData_x, trainingData_y, validationData, numEpochs, alpha, weights_filename=None):
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

    # Input layer
    inputLayer = keras.layers.Input(shape=(20,))
    
    # Encoder layer
    encoderLayer = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='encoder'
    )(inputLayer)
    
    # z-layer
    zLayer = keras.layers.Dense(
        9,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='z-layer'
    )(encoderLayer)
    
    decoderLayer = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='decoder'
    )(zLayer)
    autoencoderLayer = keras.layers.Dense(
        20,
        activation=None,
        kernel_initializer=initializer, 
        name='autoencoder'
    )(decoderLayer)
    
    classiferLayer = keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=initializer, 
        name='classifier'
    )(zLayer)
    
    outputs = [autoencoderLayer, classiferLayer]

    model = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
    model.summary()

    model.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.BinaryCrossentropy()],loss_weights=[lossWeightAlpha,1])

    if weights_filename is not None:
        model.load_weights(weights_filename)
    callbacks = []
    csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
    callbacks.append(csv_logger)
    history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, callbacks=callbacks, validation_data=validationData, verbose=0)
    
    return model

def autoencoderAll(optimizer, lossWeightAlpha, logFilename, trainingData_x, trainingData_y, numEpochs, alpha, weights_filename=None):
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

    # Input layer
    inputLayer = keras.layers.Input(shape=(20,))
    
    # Encoder layer
    encoderLayer = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='encoder'
    )(inputLayer)
    
    # z-layer
    zLayer = keras.layers.Dense(
        9,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='z-layer'
    )(encoderLayer)
    
    decoderLayer = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='decoder'
    )(zLayer)
    autoencoderLayer = keras.layers.Dense(
        20,
        activation=None,
        kernel_initializer=initializer, 
        name='autoencoder'
    )(decoderLayer)
    
    classiferLayer = keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=initializer, 
        name='classifier'
    )(zLayer)
    
    outputs = [autoencoderLayer, classiferLayer]

    model = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
    model.summary()

    model.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.BinaryCrossentropy()],loss_weights=[lossWeightAlpha,1])

    if weights_filename is not None:
        model.load_weights(weights_filename)
    callbacks = []
    csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
    callbacks.append(csv_logger)
    history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0)
    
    return model

# In the second stage, the decoder portion of the autoencoder
# is discarded, the encoder portion’s weights are frozen, and the classifier layer weights
# are reinitialized and trained using the oversampled training data.
def autoencoder_second_stage(autoencoder_validation_weights_path, optimizer, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, validationData, logFilename, lossWeightAlpha, alpha, weights_filename=None):
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

    # Input layer
    inputLayer = keras.layers.Input(shape=(20,))
    
    # Encoder layer
    encoderLayer = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='encoder'
    )(inputLayer)
    
    # z-layer
    zLayer = keras.layers.Dense(
        9,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='z-layer'
    )(encoderLayer)
    
    decoderLayer = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='decoder'
    )(zLayer)
    autoencoderLayer = keras.layers.Dense(
        20,
        activation=None,
        kernel_initializer=initializer, 
        name='autoencoder'
    )(decoderLayer)
    
    classiferLayer = keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=initializer, 
        name='classifier'
    )(zLayer)
    
    outputs = [autoencoderLayer, classiferLayer]

    autoencoder_first_stage = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
    autoencoder_first_stage.summary()

    autoencoder_first_stage.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.BinaryCrossentropy()],loss_weights=[lossWeightAlpha,1])

    autoencoder_first_stage.load_weights(autoencoder_validation_weights_path)


    initializer_ss = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

    # Input layer
    inputLayer_ss = keras.layers.Input(shape=(20,))
    
    # Encoder layer
    encoderLayerWeights = autoencoder_first_stage.layers[1].get_weights()
    encoderLayer_ss = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        trainable=False,
        weights=encoderLayerWeights,
        name='encoder'
    )(inputLayer_ss)

    # z-layer
    zLayerWeights = autoencoder_first_stage.layers[2].get_weights()
    zLayer_ss = keras.layers.Dense(
        9,
        activation=LeakyReLU(alpha=alpha),
        trainable=False,
        weights=zLayerWeights,
        name='z-layer'
    )(encoderLayer_ss)

    nonLinearLayer_ss = keras.layers.Dense(6, kernel_initializer=initializer_ss, activation=LeakyReLU(alpha=alpha))(zLayer_ss)
    
    classiferLayer_ss = keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=initializer_ss, 
        name='classifier'
    )(nonLinearLayer_ss)
    
    outputs_ss = classiferLayer_ss

    autoencoder_second_stage_model = keras.Model(inputs=inputLayer_ss, outputs=outputs_ss, name="autoencoder_model_ss")
    autoencoder_second_stage_model.summary()

    autoencoder_second_stage_model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy())
    
    
    if weights_filename is not None:
        autoencoder_second_stage_model.load_weights(weights_filename)
    callbacks = []
    csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
    callbacks.append(csv_logger)
    history = autoencoder_second_stage_model.fit(autoencoder_trainingData_x, autoencoder_trainingData_y, epochs=numEpochs, callbacks=callbacks, validation_data=validationData, verbose=0)

    return autoencoder_second_stage_model

def autoencoder_second_stage_all(autoencoder_validation_weights_path, optimizer, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, logFilename, lossWeightAlpha, alpha, weights_filename=None, seed=None, train=True):
    if seed is None:
        next_seed = getNextSeed()
    else:
        next_seed = seed
    initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=next_seed)

    # Input layer
    inputLayer = keras.layers.Input(shape=(20,))
    
    # Encoder layer
    encoderLayer = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='encoder'
    )(inputLayer)
    
    # z-layer
    zLayer = keras.layers.Dense(
        9,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='z-layer'
    )(encoderLayer)
    
    decoderLayer = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        kernel_initializer=initializer, 
        name='decoder'
    )(zLayer)
    autoencoderLayer = keras.layers.Dense(
        20,
        activation=None,
        kernel_initializer=initializer, 
        name='autoencoder'
    )(decoderLayer)
    
    classiferLayer = keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=initializer, 
        name='classifier'
    )(zLayer)
    
    outputs = [autoencoderLayer, classiferLayer]

    autoencoder_first_stage = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
    autoencoder_first_stage.summary()

    autoencoder_first_stage.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.BinaryCrossentropy()],loss_weights=[lossWeightAlpha,1])

    autoencoder_first_stage.load_weights(autoencoder_validation_weights_path)


    initializer_ss = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=(next_seed + 100))

    # Input layer
    inputLayer_ss = keras.layers.Input(shape=(20,))
    
    # Encoder layer
    encoderLayerWeights = autoencoder_first_stage.layers[1].get_weights()
    encoderLayer_ss = keras.layers.Dense(
        18,
        activation=LeakyReLU(alpha=alpha),
        trainable=False,
        weights=encoderLayerWeights,
        name='encoder'
    )(inputLayer_ss)

    # z-layer
    zLayerWeights = autoencoder_first_stage.layers[2].get_weights()
    zLayer_ss = keras.layers.Dense(
        9,
        activation=LeakyReLU(alpha=alpha),
        trainable=False,
        weights=zLayerWeights,
        name='z-layer'
    )(encoderLayer_ss)

    nonLinearLayer_ss = keras.layers.Dense(6, kernel_initializer=initializer_ss, activation=LeakyReLU(alpha=alpha))(zLayer_ss)
    
    classiferLayer_ss = keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=initializer_ss, 
        name='classifier'
    )(nonLinearLayer_ss)
    
    outputs_ss = classiferLayer_ss

    autoencoder_second_stage_model = keras.Model(inputs=inputLayer_ss, outputs=outputs_ss, name="autoencoder_model_ss")
    autoencoder_second_stage_model.summary()

    autoencoder_second_stage_model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy())
    
    
    if weights_filename is not None:
        autoencoder_second_stage_model.load_weights(weights_filename)
    if train:
        callbacks = []
        csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
        callbacks.append(csv_logger)
        history = autoencoder_second_stage_model.fit(autoencoder_trainingData_x, autoencoder_trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0)

    return autoencoder_second_stage_model