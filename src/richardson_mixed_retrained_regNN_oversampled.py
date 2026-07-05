import sys
from regression_richardson_mixed import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionRichardsonMixed()

all_trainingData_x, all_trainingData_y = reg.getFirstStageInput("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index), False)

combined_training_data_x = (richardson_trainingData_x, all_trainingData_x)
combined_training_data_y = all_trainingData_y

baseStr = "../out/richardson_mixed/retrained_regNN_oversampled_0_{}/adam_model_001_01_oversampled_0_{}".format(index, index)

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

numIterations = 5

for i in range(0, numIterations):
  logFilename = "../out/richardson_mixed/retrained_regNN_oversampled_0_{}/it_{}.csv".format(index, i)

  adam_model_001_1, adam_history_001_1 = reg.richardsonMixedNN(adamLearningRate, combined_training_data_x, combined_training_data_y, logFilename, seed=1234+i, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

  weights_path = "../out/richardson_mixed/retrained_regNN_oversampled_0_{}/it_{}.ckpt".format(index, i)
  adam_model_001_1.save_weights(weights_path)
