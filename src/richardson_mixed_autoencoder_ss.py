import sys
from regression_richardson_mixed import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionRichardsonMixed()

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)
adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

secondStageTrainingData_x_oversample, secondStageTrainingData_y_oversample = reg.getFirstStageInput("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
secondStageValidationData_oversample = reg.getValidationData("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index), False)
richardson_validationData = reg.getValidationDataRichardson("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index), False)

combined_training_data_x = (richardson_trainingData_x, secondStageTrainingData_x_oversample)
combined_training_data_y = secondStageTrainingData_y_oversample
combined_validation_data = ((richardson_validationData[0], secondStageValidationData_oversample[0]), richardson_validationData[1])

autoencoder_trainingData_x = combined_training_data_x
autoencoder_trainingData_y = combined_training_data_y
autoencoder_validationData = combined_validation_data

autoencoder_validation_weights_path = "../out/richardson_mixed/autoencoder_retrained/autoencoder_validation_training.ckpt"

logFilename = "../out/richardson_mixed/autoencoder_ss_0_{}/autoencoder_ss_0_{}_validation_training.csv".format(index, index)

autoencoder_model = reg.richardsonMixed_autoencoder_second_stage(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilename, alpha)

autoencoder_validation_weights_path = "../out/richardson_mixed/autoencoder_ss_0_{}/autoencoder_ss_0_{}_validation_training.ckpt".format(index, index)
autoencoder_model.save_weights(autoencoder_validation_weights_path)
