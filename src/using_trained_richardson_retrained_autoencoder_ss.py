import sys
from regression_trained_richardson import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionTrainedRichardson()

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

secondStageTrainingData_x_oversample, secondStageTrainingData_y_oversample = reg.getFirstStageInput("../res/trained/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))

autoencoder_trainingData_x = secondStageTrainingData_x_oversample
autoencoder_trainingData_y = secondStageTrainingData_y_oversample

autoencoder_validation_weights_path = "../out/using_trained_richardson/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"

numIterations = 5
for i in range(0, numIterations):
    adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)
    adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

    logFilename = "../out/using_trained_richardson/retrained_autoencoder_ss_0_{}/it_{}.csv".format(index, i)

    autoencoder_second_stage_model = reg.autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, logFilename, alpha, seed=1234+i)

    autoencoder_validation_weights_path_ss = "../out/using_trained_richardson/retrained_autoencoder_ss_0_{}/it_{}.ckpt".format(index, i)
    autoencoder_second_stage_model.save_weights(autoencoder_validation_weights_path_ss)
