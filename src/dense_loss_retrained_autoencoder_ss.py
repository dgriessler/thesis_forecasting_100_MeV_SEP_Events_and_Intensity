import sys
from regression import *

denseLossAlpha = float(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = Regression()

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

autoencoder_trainingData_x = all_trainingData_x
autoencoder_trainingData_y = all_trainingData_y

numIterations = 5
for i in range(0, numIterations):
    autoencoder_validation_weights_path = "../out/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"

    adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)
    adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

    logFilename = "../out/denseLoss_retrained_autoencoder_ss/alpha_{:.2f}_it_{}.csv".format(denseLossAlpha, i)

    autoencoder_model = reg.denseLoss_autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, logFilename, denseLossAlpha, alpha, 1234+i)

    save_weights_path = "../out/denseLoss_retrained_autoencoder_ss/alpha_{:.2f}_it_{}.ckpt".format(denseLossAlpha, i)
    autoencoder_model.save_weights(save_weights_path)
