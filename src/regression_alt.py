import csv
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
from base_func_alt import *

class RegressionAlt(object):
    def __init__(self):
        self.lossWeightAlpha = 0.570642544

    def getFirstStageInput(self, csvFile):
        data = readCSVFile(csvFile)
    
        numElements = len(data)
        inputData_x = np.empty((numElements, 19), dtype=np.float64)
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
        
            inputData_y[i][0] = float(elem["100MeV_peak_intensity_ln"])
    
        return (inputData_x, inputData_y)

    def getTargets(self, csvFile):
        data = readCSVFile(csvFile)
        numElements = len(data)

        keysData = np.empty((numElements), dtype=np.intc)
        keysData_i = 0
        for i in range(0, numElements):
            elem = data[i]
        
            try:
                keysData[keysData_i] = int(elem["target"])
                keysData_i = keysData_i + 1
            except:
                print("ERROR: INVALID DATA: {}".format(elem), file=sys.stderr)
        keysData = np.resize(keysData, (keysData_i))
        print("Target New size x: {}".format(keysData.shape))

        return keysData

    def getValidationData(self, csvFile):
        # Validation data should be tuples (x_val, y_val) of Numpy arrays or tensors
        data_x, data_y = self.getFirstStageInput(csvFile)

        validationData = (data_x, data_y)
    
        return validationData

    def getFirstStageInputRichardson(self, csvFile, useSynData = False):
        data = readCSVFile(csvFile)
    
        numElements = len(data)
        inputData_x = np.empty((numElements, 2), dtype=np.float64)
        inputData_y = np.empty((numElements, 1), dtype=np.float64)
    
        yKey = "100MeV_peak_intensity_ln"
        if useSynData:
            yKey = "expected_richardson_ln"

        input_i = 0
        for i in range(0, numElements):
            elem = data[i]
        
            try:
                inputData_x[i][0] = float(elem["donki_speed_unnormalized"])
                inputData_x[i][1] = float(elem["connection_angle_degrees_phi_2_solar_wind_sq_div"])
        
                inputData_y[i][0] = float(elem[yKey])

                input_i = input_i + 1
            except:
                print("ERROR: INVALID DATA: {}".format(elem), file=sys.stderr)
        inputData_x = np.resize(inputData_x, (input_i, 2))
        inputData_y = np.resize(inputData_y, (input_i, 1))
        print("Input New size x: {}".format(inputData_x.shape))
        print("Input New size y: {}".format(inputData_y.shape))
    
        return (inputData_x, inputData_y)

    def getValidationDataRichardson(self, csvFile, useSynData=True):
        # Validation data should be tuples (x_val, y_val) of Numpy arrays or tensors
        data_x, data_y = self.getFirstStageInputRichardson(csvFile, useSynData=useSynData)

        validationData = (data_x, data_y)
    
        return validationData

    def getFirstStageInputRichardsonLearnSigma(self, csvFile):
        data = readCSVFile(csvFile)
        learned_richardson = readCSVFile("../res/trained/richardson_learning_w_exp_w_v.csv")
        learned_w_exp = float(learned_richardson[0]["w_exp"])
        learned_w_v = float(learned_richardson[0]["w_v"])
    
        numElements = len(data)
        inputData_x = np.empty((numElements, 2), dtype=np.float64)
        inputData_y = np.empty((numElements, 1), dtype=np.float64)
    
        yKey = "100MeV_peak_intensity_ln"

        input_i = 0
        for i in range(0, numElements):
            elem = data[i]
        
            try:
                inputData_x[i][0] = math.log(learned_w_exp) + learned_w_v * float(elem["donki_speed"])
                inputData_x[i][1] = -1 * math.pow(float(elem["connection_angle_degrees"]), 2) / 2
        
                inputData_y[i][0] = float(elem[yKey])

                input_i = input_i + 1
            except:
                print("ERROR: INVALID DATA: {}".format(elem), file=sys.stderr)
        inputData_x = np.resize(inputData_x, (input_i, 2))
        inputData_y = np.resize(inputData_y, (input_i, 1))
        print("Input New size x: {}".format(inputData_x.shape))
        print("Input New size y: {}".format(inputData_y.shape))
    
        return (inputData_x, inputData_y)

    def getValidationDataRichardsonLearnSigma(self, csvFile):
        # Validation data should be tuples (x_val, y_val) of Numpy arrays or tensors
        data_x, data_y = self.getFirstStageInputRichardsonLearnSigma(csvFile)

        validationData = (data_x, data_y)
    
        return validationData

    def getFirstStageInputRichardsonLearnWExpWVSigma(self, csvFile):
        data = readCSVFile(csvFile)
    
        numElements = len(data)
        inputData_x = np.empty((numElements, 2), dtype=np.float64)
        inputData_y = np.empty((numElements, 1), dtype=np.float64)
    
        yKey = "100MeV_peak_intensity_ln"

        input_i = 0
        for i in range(0, numElements):
            elem = data[i]
        
            try:
                inputData_x[i][0] = float(elem["donki_speed"])
                inputData_x[i][1] = -1 * math.pow(float(elem["connection_angle_degrees"]), 2) / 2
        
                inputData_y[i][0] = float(elem[yKey])

                input_i = input_i + 1
            except:
                print("ERROR: INVALID DATA: {}".format(elem), file=sys.stderr)
        inputData_x = np.resize(inputData_x, (input_i, 2))
        inputData_y = np.resize(inputData_y, (input_i, 1))
        print("Input New size x: {}".format(inputData_x.shape))
        print("Input New size y: {}".format(inputData_y.shape))
    
        return (inputData_x, inputData_y)

    def getValidationDataRichardsonLearnWExpWVSigma(self, csvFile):
        # Validation data should be tuples (x_val, y_val) of Numpy arrays or tensors
        data_x, data_y = self.getFirstStageInputRichardsonLearnWExpWVSigma(csvFile)

        validationData = (data_x, data_y)
    
        return validationData

    def getFirstStageInputRichardsonLearnWVSigma(self, csvFile):
        data = readCSVFile(csvFile)
    
        numElements = len(data)
        inputData_x = np.empty((numElements, 2), dtype=np.float64)
        inputData_y = np.empty((numElements, 1), dtype=np.float64)
    
        yKey = "100MeV_peak_intensity_ln"

        input_i = 0
        for i in range(0, numElements):
            elem = data[i]
        
            try:
                inputData_x[i][0] = float(elem["donki_speed"])
                inputData_x[i][1] = -1 * math.pow(float(elem["connection_angle_degrees"]), 2) / 2
        
                inputData_y[i][0] = float(elem[yKey])

                input_i = input_i + 1
            except:
                print("ERROR: INVALID DATA: {}".format(elem), file=sys.stderr)
        inputData_x = np.resize(inputData_x, (input_i, 2))
        inputData_y = np.resize(inputData_y, (input_i, 1))
        print("Input New size x: {}".format(inputData_x.shape))
        print("Input New size y: {}".format(inputData_y.shape))
    
        return (inputData_x, inputData_y)

    def getValidationDataRichardsonLearnWVSigma(self, csvFile):
        # Validation data should be tuples (x_val, y_val) of Numpy arrays or tensors
        data_x, data_y = self.getFirstStageInputRichardsonLearnWVSigma(csvFile)

        validationData = (data_x, data_y)
    
        return validationData

    def getFixedW0Richardson(self, csvFile):
        data = readCSVFile(csvFile)
    
        numElements = len(data)
        # Set to average of the ln intensity
        w_exp = 0
    
        yKey = "100MeV_peak_intensity_ln"

        for i in range(0, numElements):
            elem = data[i]

            w_exp = w_exp + float(elem[yKey]) / numElements
    
        return w_exp

    def regNNValidation(self, adamLearningRate, trainingData_x, trainingData_y, validationData, logFilename, adamEpsilon, alpha, numEpochs=3000, weights_filename=None):
        print("Adam Learning Rate: {:.5f}. Adam epsilon: {:.5f}. Num epochs: {:}. Alpha: {:.5f}.".format(adamLearningRate, adamEpsilon, numEpochs, alpha))

        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())
        # Input layer
        inputLayer = keras.Input(shape=(19,))

        # Hidden layer
        encoderLayer = keras.layers.Dense(18, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(inputLayer)

        zLayer = keras.layers.Dense(9, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(encoderLayer)

        # Output Layer
        outputLayer = layers.Dense(1, activation=None, kernel_initializer=initializer)(zLayer)

        model = keras.Model(inputs=inputLayer, outputs=outputLayer, name="adapted_rRT_" + str(adamLearningRate))
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.MeanSquaredError())
        # strategy end

        if weights_filename is not None:
            model.load_weights(weights_filename)
            history = None
        else:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
            history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, validation_data=validationData, callbacks=callbacks, verbose=0)
    
        return (model, history)

    def regNNValidationCont(self, model, trainingData_x, trainingData_y, validationData, logFilename, numEpochs=3000, weights_filename=None):
        if weights_filename is not None:
            model.load_weights(weights_filename)
            history = None
        else:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
            history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, validation_data=validationData, callbacks=callbacks, verbose=0)
    
        return (model, history)

    def regNN(self, adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, seed, adamEpsilon, alpha, numEpochs=150, weights_filename=None):
        print("Adam Learning Rate: {:.5f}. Adam epsilon: {:.5f}. Num epochs: {:}. Alpha: {:.5f}.".format(adamLearningRate, adamEpsilon, numEpochs, alpha))
        
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
        # Input layer
        inputLayer = keras.Input(shape=(19,))

        # Hidden layer
        encoderLayer = keras.layers.Dense(18, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(inputLayer)

        zLayer = keras.layers.Dense(9, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(encoderLayer)

        # Output Layer
        outputLayer = layers.Dense(1, activation=None, kernel_initializer=initializer)(zLayer)

        model = keras.Model(inputs=inputLayer, outputs=outputLayer, name="adapted_rRT_" + str(adamLearningRate))
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.MeanSquaredError())
    
        if weights_filename is not None:
            model.load_weights(weights_filename)
            history = None
        else:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
            history = model.fit(all_trainingData_x, all_trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0)
    
        return (model, history)

    def rRTValidation(self, adamLearningRate, model, trainingData_x, trainingData_y, validationData, logFilename, adamEpsilon, alpha, numEpochs=3000, weights_filename=None, do_train=True):
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Freeze all but the last layer

        # Input layer
        inputs = model.inputs
        x = model.output

        nonLinearLayer = keras.layers.Dense(6, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(x)

        # Reinitialize classifier weights
        outputLayerRetrain = layers.Dense(1, activation=None, kernel_initializer=initializer)(nonLinearLayer)

        modelRetrain = keras.Model(inputs=inputs, outputs=outputLayerRetrain, name="adapted_rRT_retrain")
        modelRetrain.summary()

        # Optimizer = model.optimizer
        # Loss = Mean Squared Error
        modelRetrain.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.MeanSquaredError())
    
        if weights_filename is not None:
            historyRetrain = None
            modelRetrain.load_weights(weights_filename)
        if do_train:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
    
            historyRetrain = modelRetrain.fit(trainingData_x, trainingData_y, epochs=numEpochs, validation_data=validationData, callbacks=callbacks, verbose=0)
    
        return (modelRetrain, historyRetrain)

    def rRT(self, adamLearningRate, model, all_trainingData_x, all_trainingData_y, logFilename, seed, adamEpsilon, alpha, numEpochs=3000, weights_filename=None):
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

        # Input layer
        inputs = model.inputs
        x = model.output

        nonLinearLayer = keras.layers.Dense(6, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(x)

        # Reinitialize classifier weights
        outputLayerRetrain = layers.Dense(1, activation=None, kernel_initializer=initializer)(nonLinearLayer)

        modelRetrain = keras.Model(inputs=inputs, outputs=outputLayerRetrain, name="adapted_rRT_all_retrain")
        modelRetrain.summary()

        # Optimizer = model.optimizer
        # Loss = Mean Squared Error
        modelRetrain.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.MeanSquaredError())
    
        if weights_filename is not None:
            historyRetrain = None
            modelRetrain.load_weights(weights_filename)
        else:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
        
            historyRetrain = modelRetrain.fit(all_trainingData_x, all_trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0)
    
        return (modelRetrain, historyRetrain)

    def autoencoder_finding_alpha_auto(self, optimizer, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilename, numEpochs, weights_filename, alpha):
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Autoencoder
        # Input layer
        inputLayerFirst = keras.layers.Input(shape=(19,))

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
            19,
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

        return modelFirst

    def autoencoder_finding_alpha_class(self, optimizerClassifier, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilenameClassifier, numEpochs, alpha, weights_filename_classifier):
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Classifier
        # Input layer
        inputLayerSecond = keras.layers.Input(shape=(19,))
    
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
            activation=None,
            kernel_initializer=initializer, 
            name='classifier'
        )(zLayerSecond)

        outputSecond = classiferLayerSecond

   
        modelSecond = keras.Model(inputs=inputLayerSecond, outputs=outputSecond, name="autoencoder_classifier_model")
        modelSecond.summary()

        modelSecond.compile(optimizer=optimizerClassifier, loss=keras.losses.MeanSquaredError())

        callbacksClassifier = []
        csv_logger_classifier = keras.callbacks.CSVLogger(logFilenameClassifier, append=True, separator=';')
        callbacksClassifier.append(csv_logger_classifier)
        historySecond = modelSecond.fit(autoencoder_trainingData_x, autoencoder_trainingData_y, epochs=numEpochs, callbacks=callbacksClassifier, validation_data=autoencoder_validationData, verbose=0)

        modelSecond.save_weights(weights_filename_classifier)

        return modelSecond

    def autoencoder(self, optimizer, logFilename, trainingData_x, trainingData_y, validationData, numEpochs, alpha, weights_filename=None):
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Input layer
        inputLayer = keras.layers.Input(shape=(19,))
    
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
            19,
            activation=None,
            kernel_initializer=initializer, 
            name='autoencoder'
        )(decoderLayer)
    
        classiferLayer = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=initializer, 
            name='classifier'
        )(zLayer)
    
        outputs = [autoencoderLayer, classiferLayer]

        model = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
        model.summary()

        model.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.MeanSquaredError()],loss_weights=[self.lossWeightAlpha,1])

        if weights_filename is not None:
            model.load_weights(weights_filename)
        callbacks = []
        csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
        callbacks.append(csv_logger)
        history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, callbacks=callbacks, validation_data=validationData, verbose=0)
    
        return model

    def autoencoderAll(self, optimizer, logFilename, trainingData_x, trainingData_y, numEpochs, alpha, weights_filename=None):
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Input layer
        inputLayer = keras.layers.Input(shape=(19,))
    
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
            19,
            activation=None,
            kernel_initializer=initializer, 
            name='autoencoder'
        )(decoderLayer)
    
        classiferLayer = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=initializer, 
            name='classifier'
        )(zLayer)
    
        outputs = [autoencoderLayer, classiferLayer]

        model = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
        model.summary()

        model.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.MeanSquaredError()],loss_weights=[self.lossWeightAlpha,1])

        if weights_filename is not None:
            model.load_weights(weights_filename)
        callbacks = []
        csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
        callbacks.append(csv_logger)
        history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0)
    
        return model

    def autoencoder_second_stage(self, autoencoder_validation_weights_path, optimizer, optimizer_ss, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, validationData, logFilename, alpha, weights_filename=None):
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Input layer
        inputLayer = keras.layers.Input(shape=(19,))
    
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
            19,
            activation=None,
            kernel_initializer=initializer, 
            name='autoencoder'
        )(decoderLayer)
    
        classiferLayer = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=initializer, 
            name='classifier'
        )(zLayer)
    
        outputs = [autoencoderLayer, classiferLayer]

        autoencoder_first_stage = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
        autoencoder_first_stage.summary()

        autoencoder_first_stage.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.MeanSquaredError()],loss_weights=[self.lossWeightAlpha,1])

        autoencoder_first_stage.load_weights(autoencoder_validation_weights_path)


        initializer_ss = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Input layer
        inputLayer_ss = keras.layers.Input(shape=(19,))
    
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
            activation=None,
            kernel_initializer=initializer_ss, 
            name='classifier'
        )(nonLinearLayer_ss)
    
        outputs_ss = classiferLayer_ss

        autoencoder_second_stage_model = keras.Model(inputs=inputLayer_ss, outputs=outputs_ss, name="autoencoder_model_ss")
        autoencoder_second_stage_model.summary()

        autoencoder_second_stage_model.compile(optimizer=optimizer_ss, loss=keras.losses.MeanSquaredError())
    
    
        if weights_filename is not None:
            autoencoder_second_stage_model.load_weights(weights_filename)
        callbacks = []
        csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
        callbacks.append(csv_logger)
        history = autoencoder_second_stage_model.fit(autoencoder_trainingData_x, autoencoder_trainingData_y, epochs=numEpochs, callbacks=callbacks, validation_data=validationData, verbose=0)

        return autoencoder_second_stage_model

    def autoencoder_second_stage_all(self, autoencoder_validation_weights_path, optimizer, optimizer_ss, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, logFilename, alpha, seed, weights_filename=None, train=True):
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

        # Input layer
        inputLayer = keras.layers.Input(shape=(19,))
    
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
            19,
            activation=None,
            kernel_initializer=initializer, 
            name='autoencoder'
        )(decoderLayer)
    
        classiferLayer = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=initializer, 
            name='classifier'
        )(zLayer)
    
        outputs = [autoencoderLayer, classiferLayer]

        autoencoder_first_stage = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
        autoencoder_first_stage.summary()

        autoencoder_first_stage.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.MeanSquaredError()],loss_weights=[self.lossWeightAlpha,1])

        autoencoder_first_stage.load_weights(autoencoder_validation_weights_path)


        initializer_ss = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

        # Input layer
        inputLayer_ss = keras.layers.Input(shape=(19,))
    
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
            activation=None,
            kernel_initializer=initializer_ss, 
            name='classifier'
        )(nonLinearLayer_ss)
    
        outputs_ss = classiferLayer_ss

        autoencoder_second_stage_model = keras.Model(inputs=inputLayer_ss, outputs=outputs_ss, name="autoencoder_model_ss")
        autoencoder_second_stage_model.summary()

        autoencoder_second_stage_model.compile(optimizer=optimizer_ss, loss=keras.losses.MeanSquaredError())
    
    
        if weights_filename is not None:
            autoencoder_second_stage_model.load_weights(weights_filename)
        if train:
            callbacks = []
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks.append(csv_logger)
            history = autoencoder_second_stage_model.fit(autoencoder_trainingData_x, autoencoder_trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0)

        return autoencoder_second_stage_model

    def denseLossGetLossFunction(self, denseLossAlpha):
        if math.isclose(denseLossAlpha, 0.0):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_0
        elif math.isclose(denseLossAlpha, 0.1):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_1
        elif math.isclose(denseLossAlpha, 0.2):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_2
        elif math.isclose(denseLossAlpha, 0.3):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_3
        elif math.isclose(denseLossAlpha, 0.4):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_4
        elif math.isclose(denseLossAlpha, 0.5):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_5
        elif math.isclose(denseLossAlpha, 0.6):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_6
        elif math.isclose(denseLossAlpha, 0.7):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_7
        elif math.isclose(denseLossAlpha, 0.8):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_8
        elif math.isclose(denseLossAlpha, 0.9):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_9
        elif math.isclose(denseLossAlpha, 1.0):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha1_0
        elif math.isclose(denseLossAlpha, 1.5):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha1_5
        elif math.isclose(denseLossAlpha, 2.0):
            print("DENSELOSSALPHA: {}".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha2_0
        else:
            print("UNRECOGNIZED DENSELOSSALPHA: {}. DEFAULTING TO 0.0".format(denseLossAlpha))
            lossFunc = DenseLossWithMeanSquaredErrorAndAlpha0_0
        return lossFunc

    def denseLossNNModelValidation(self, adamLearningRate, trainingData_x, trainingData_y, validationData, logFilename, denseLossAlpha, adamEpsilon, alpha, doTrain, numEpochs=3000, weights_filename=None):
        print("Adam Learning Rate: {:.5f}. Adam epsilon: {:.5f}. Num epochs: {:}. Alpha: {:.5f}.".format(adamLearningRate, adamEpsilon, numEpochs, alpha))

        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())
        # Input layer
        inputLayer = keras.Input(shape=(19,))

        # Hidden layer
        encoderLayer = keras.layers.Dense(18, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(inputLayer)

        zLayer = keras.layers.Dense(9, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(encoderLayer)

        # Output Layer
        outputLayer = layers.Dense(1, activation=None, kernel_initializer=initializer)(zLayer)

        model = keras.Model(inputs=inputLayer, outputs=outputLayer, name="adapted_rRT_" + str(adamLearningRate))
        model.summary()

        lossFunc = self.denseLossGetLossFunction(denseLossAlpha)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=lossFunc)
        # strategy end

        if weights_filename is not None:
            model.load_weights(weights_filename)
            history = None
        if doTrain:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
            history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, validation_data=validationData, callbacks=callbacks, verbose=0, batch_size=179)
    
        return (model, history)


    def denseLossNNModel(self, adamLearningRate, trainingData_x, trainingData_y, logFilename, seed, denseLossAlpha, adamEpsilon, alpha, numEpochs=3000, weights_filename=None):
        print("Adam Learning Rate: {:.5f}. Adam epsilon: {:.5f}. Num epochs: {:}. Alpha: {:.5f}.".format(adamLearningRate, adamEpsilon, numEpochs, alpha))

        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
        # Input layer
        inputLayer = keras.Input(shape=(19,))

        # Hidden layer
        encoderLayer = keras.layers.Dense(18, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(inputLayer)

        zLayer = keras.layers.Dense(9, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(encoderLayer)

        # Output Layer
        outputLayer = layers.Dense(1, activation=None, kernel_initializer=initializer)(zLayer)

        model = keras.Model(inputs=inputLayer, outputs=outputLayer, name="adapted_rRT_" + str(adamLearningRate))
        model.summary()

        lossFunc = self.denseLossGetLossFunction(denseLossAlpha)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=lossFunc)
        # strategy end

        if weights_filename is not None:
            model.load_weights(weights_filename)
            history = None
        else:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
            history = model.fit(trainingData_x, trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0, batch_size=168)
    
        return (model, history)

    def denseLoss_rRTValidation(self, adamLearningRate, model, trainingData_x, trainingData_y, validationData, logFilename, denseLossAlpha, adamEpsilon, alpha, numEpochs=3000, weights_filename=None, do_train=True):
        # model should be the regular NN at 0 percent oversampling from 1 of the 5 runs
        # Then we discard the regressor and replace it with a dense loss function, don't need oversampling because dense loss does reweighting

        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Freeze all but the last layer

        # Input layer
        inputs = model.inputs
        x = model.output

        nonLinearLayer = keras.layers.Dense(6, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(x)

        # Reinitialize classifier weights
        outputLayerRetrain = layers.Dense(1, activation=None, kernel_initializer=initializer)(nonLinearLayer)

        modelRetrain = keras.Model(inputs=inputs, outputs=outputLayerRetrain, name="adapted_rRT_retrain")
        modelRetrain.summary()

        # Optimizer = model.optimizer
        lossFunc = self.denseLossGetLossFunction(denseLossAlpha)
        modelRetrain.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=lossFunc)
    
        if weights_filename is not None:
            historyRetrain = None
            modelRetrain.load_weights(weights_filename)
        if do_train:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
    
            historyRetrain = modelRetrain.fit(trainingData_x, trainingData_y, epochs=numEpochs, validation_data=validationData, callbacks=callbacks, verbose=0, batch_size=179)
    
        return (modelRetrain, historyRetrain)

    def denseLoss_rRT(self, adamLearningRate, model, all_trainingData_x, all_trainingData_y, logFilename, denseLossAlpha, seed, adamEpsilon, alpha, numEpochs=3000, weights_filename=None):
        # model should be the regular NN at 0 percent oversampling from 1 of the 5 runs
        # Then we discard the regressor and replace it with a dense loss function, don't need oversampling because dense loss does reweighting

        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

        # Input layer
        inputs = model.inputs
        x = model.output

        nonLinearLayer = keras.layers.Dense(6, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(x)

        # Reinitialize classifier weights
        outputLayerRetrain = layers.Dense(1, activation=None, kernel_initializer=initializer)(nonLinearLayer)

        modelRetrain = keras.Model(inputs=inputs, outputs=outputLayerRetrain, name="adapted_rRT_all_retrain")
        modelRetrain.summary()

        # Optimizer = model.optimizer
        lossFunc = self.denseLossGetLossFunction(denseLossAlpha)
        modelRetrain.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=lossFunc)
    
        if weights_filename is not None:
            historyRetrain = None
            modelRetrain.load_weights(weights_filename)
        else:
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks = []
            callbacks.append(csv_logger)
        
            historyRetrain = modelRetrain.fit(all_trainingData_x, all_trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0, batch_size=168)
    
        return (modelRetrain, historyRetrain)

    def denseLoss_autoencoder_second_stage(self, autoencoder_validation_weights_path, optimizer, optimizer_ss, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, validationData, logFilename, denseLossAlpha, alpha, weights_filename=None):
        # autoencoder_validation_weights_path should be autoencoder stage 1 from regular NN on 1 of its 5 runs
        # we'll throw away the regressor and do dense loss in its place, no need for oversampled data since dense loss provides the weighting

        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Input layer
        inputLayer = keras.layers.Input(shape=(19,))
    
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
            19,
            activation=None,
            kernel_initializer=initializer, 
            name='autoencoder'
        )(decoderLayer)
    
        classiferLayer = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=initializer, 
            name='classifier'
        )(zLayer)
    
        outputs = [autoencoderLayer, classiferLayer]

        autoencoder_first_stage = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
        autoencoder_first_stage.summary()

        autoencoder_first_stage.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.MeanSquaredError()],loss_weights=[self.lossWeightAlpha,1])

        autoencoder_first_stage.load_weights(autoencoder_validation_weights_path)


        initializer_ss = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=getNextSeed())

        # Input layer
        inputLayer_ss = keras.layers.Input(shape=(19,))
    
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
            activation=None,
            kernel_initializer=initializer_ss, 
            name='classifier'
        )(nonLinearLayer_ss)
    
        outputs_ss = classiferLayer_ss

        autoencoder_second_stage_model = keras.Model(inputs=inputLayer_ss, outputs=outputs_ss, name="autoencoder_model_ss")
        autoencoder_second_stage_model.summary()

        lossFunc = self.denseLossGetLossFunction(denseLossAlpha)

        autoencoder_second_stage_model.compile(optimizer=optimizer_ss, loss=lossFunc)
    
    
        if weights_filename is not None:
            autoencoder_second_stage_model.load_weights(weights_filename)
        callbacks = []
        csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
        callbacks.append(csv_logger)
        history = autoencoder_second_stage_model.fit(autoencoder_trainingData_x, autoencoder_trainingData_y, epochs=numEpochs, callbacks=callbacks, validation_data=validationData, verbose=0, batch_size=179)

        return autoencoder_second_stage_model

    def denseLoss_autoencoder_second_stage_all(self, autoencoder_validation_weights_path, optimizer, optimizer_ss, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, logFilename, denseLossAlpha, alpha, seed, weights_filename=None, train=True):
        # autoencoder_validation_weights_path should be autoencoder stage 1 from regular NN on 1 of its 5 runs
        # we'll throw away the regressor and do dense loss in its place, no need for oversampled data since dense loss provides the weighting

        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

        # Input layer
        inputLayer = keras.layers.Input(shape=(19,))
    
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
            19,
            activation=None,
            kernel_initializer=initializer, 
            name='autoencoder'
        )(decoderLayer)
    
        classiferLayer = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=initializer, 
            name='classifier'
        )(zLayer)
    
        outputs = [autoencoderLayer, classiferLayer]

        autoencoder_first_stage = keras.Model(inputs=inputLayer, outputs=outputs, name="autoencoder_model")
        autoencoder_first_stage.summary()

        autoencoder_first_stage.compile(optimizer=optimizer, loss=[keras.losses.MeanSquaredError(),keras.losses.MeanSquaredError()],loss_weights=[self.lossWeightAlpha,1])

        autoencoder_first_stage.load_weights(autoencoder_validation_weights_path)


        initializer_ss = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

        # Input layer
        inputLayer_ss = keras.layers.Input(shape=(19,))
    
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
            activation=None,
            kernel_initializer=initializer_ss, 
            name='classifier'
        )(nonLinearLayer_ss)
    
        outputs_ss = classiferLayer_ss

        autoencoder_second_stage_model = keras.Model(inputs=inputLayer_ss, outputs=outputs_ss, name="autoencoder_model_ss")
        autoencoder_second_stage_model.summary()

        lossFunc = self.denseLossGetLossFunction(denseLossAlpha)

        autoencoder_second_stage_model.compile(optimizer=optimizer_ss, loss=lossFunc)
    
    
        if weights_filename is not None:
            autoencoder_second_stage_model.load_weights(weights_filename)
        if train:
            callbacks = []
            csv_logger = keras.callbacks.CSVLogger(logFilename, append=True, separator=';')
            callbacks.append(csv_logger)
            history = autoencoder_second_stage_model.fit(autoencoder_trainingData_x, autoencoder_trainingData_y, epochs=numEpochs, callbacks=callbacks, verbose=0, batch_size=168)

        return autoencoder_second_stage_model
