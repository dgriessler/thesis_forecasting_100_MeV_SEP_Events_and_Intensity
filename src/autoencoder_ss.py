import sys
from regression import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = Regression()

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageAllTraining.csv")


adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)
adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

secondStageTrainingData_x_oversample, secondStageTrainingData_y_oversample = reg.getFirstStageInput("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
secondStageValidationData_oversample = reg.getValidationData("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))

autoencoder_trainingData_x = secondStageTrainingData_x_oversample
autoencoder_trainingData_y = secondStageTrainingData_y_oversample
autoencoder_validationData = secondStageValidationData_oversample

autoencoder_validation_weights_path = "../out/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"

logFilename = "../out/autoencoder_ss_0_{}/autoencoder_ss_0_{}_validation_training.csv".format(index, index)

autoencoder_model = reg.autoencoder_second_stage(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilename, alpha)

autoencoder_validation_weights_path = "../out/autoencoder_ss_0_{}/autoencoder_ss_0_{}_validation_training.ckpt".format(index, index)
autoencoder_model.save_weights(autoencoder_validation_weights_path)
