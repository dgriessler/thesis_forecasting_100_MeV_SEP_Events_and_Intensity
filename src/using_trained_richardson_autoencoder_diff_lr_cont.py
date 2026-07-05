import sys
from regression_trained_richardson import *

reg = RegressionTrainedRichardson()

numEpochs = int(sys.argv[1])

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/trained/firstStageTraining.csv")
validationData = reg.getValidationData("../res/trained/firstStageValidation.csv")


# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/trained/firstStageAllTraining.csv")


adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

autoencoder_trainingData_x = trainingData_x
autoencoder_trainingData_y = [trainingData_x, trainingData_y]
autoencoder_validationData = (validationData[0], [validationData[0], validationData[1]])

logFilename = "../out/using_trained_richardson/autoencoder/autoencoder_validation_training.csv"

autoencoder_validation_weights_path = "../out/using_trained_richardson/autoencoder/autoencoder_validation_training.ckpt"

autoencoder_model = reg.autoencoder(adamOptimizer, logFilename, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, numEpochs, alpha, autoencoder_validation_weights_path)
keras.utils.plot_model(autoencoder_model, to_file='../out/using_trained_richardson/autoencoder/autoencoder_model.png', show_shapes=True, show_layer_names=True)


autoencoder_model.save_weights(autoencoder_validation_weights_path)
