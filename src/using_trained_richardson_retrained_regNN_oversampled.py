import sys
from regression_trained_richardson import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionTrainedRichardson()

all_trainingData_x, all_trainingData_y = reg.getFirstStageInput("../res/trained/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))

baseStr = "../out/using_trained_richardson/retrained_regNN_oversampled_0_{}/adam_model_001_01_oversampled_0_{}".format(index, index)

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

numIterations = 5

for i in range(0, numIterations):
  logFilename = "../out/using_trained_richardson/retrained_regNN_oversampled_0_{}/it_{}.csv".format(index, i)

  adam_model_001_1, adam_history_001_1 = reg.regNN(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, seed=1234+i, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

  weights_path = "../out/using_trained_richardson/retrained_regNN_oversampled_0_{}/it_{}.ckpt".format(index, i)
  adam_model_001_1.save_weights(weights_path)
