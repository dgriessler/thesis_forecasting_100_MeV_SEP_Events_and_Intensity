import sys
from regression_richardson_mixed import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionRichardsonMixed()

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

secondStageTrainingData_x_oversample, secondStageTrainingData_y_oversample = reg.getFirstStageInput("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index), False)

combined_training_data_x = (richardson_trainingData_x, secondStageTrainingData_x_oversample)
combined_training_data_y = secondStageTrainingData_y_oversample

autoencoder_trainingData_x = combined_training_data_x
autoencoder_trainingData_y = combined_training_data_y

for i in range(0, 5):
    adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)
    adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

    autoencoder_validation_weights_path = "../out/richardson_mixed/autoencoder_retrained/autoencoder_validation_training.ckpt"

    logFilename = "../out/richardson_mixed/retrained_autoencoder_ss_0_{}/it_{}.csv".format(index, i)

    autoencoder_model = reg.richardsonMixed_autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, numEpochs, autoencoder_trainingData_x, autoencoder_trainingData_y, logFilename, alpha, 1234+i)

    save_weights_filename = "../out/richardson_mixed/retrained_autoencoder_ss_0_{}/it_{}.ckpt".format(index, i)
    autoencoder_model.save_weights(save_weights_filename)
