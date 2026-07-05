import sys
from regression import *

denseLossAlpha = float(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = Regression()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = reg.getValidationData("../res/gen/firstStageValidation.csv")


adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)
adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

autoencoder_trainingData_x = trainingData_x
autoencoder_trainingData_y = trainingData_y
autoencoder_validationData = validationData

autoencoder_validation_weights_path = "../out/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"

logFilename = "../out/denseLoss_autoencoder_ss/alpha_{:.2f}.csv".format(denseLossAlpha)

save_weights_path = "../out/denseLoss_autoencoder_ss/alpha_{:.2f}.ckpt".format(denseLossAlpha)
autoencoder_model = reg.denseLoss_autoencoder_second_stage(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilename, denseLossAlpha, alpha, weights_filename=save_weights_path)

autoencoder_model.save_weights(save_weights_path)
