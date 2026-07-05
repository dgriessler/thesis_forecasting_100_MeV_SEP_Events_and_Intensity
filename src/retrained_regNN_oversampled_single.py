import sys
from regression import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])
i = int(sys.argv[3])

reg = Regression()

all_trainingData_x_0_1, all_trainingData_y_0_1 = reg.getFirstStageInput("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))

testData_x_0_1, testData_y_0_1 = reg.getFirstStageInput("../res/gen/secondStageOversampleTest_percentSEP_0.{}.csv".format(index))
testData_target_0_1 = reg.getTargets("../res/gen/secondStageOversampleTest_percentSEP_0.{}.csv".format(index))

baseStr = "../out/retrained_regNN_oversampled_0_{}/adam_model_001_01_oversampled_0_{}".format(index, index)

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

numIterations = 5

logFilename = "../out/retrained_regNN_oversampled_0_{}/it_{}.csv".format(index, i)

adam_model_001_1, adam_history_001_1 = reg.regNN(adamLearningRate, all_trainingData_x_0_1, all_trainingData_y_0_1, logFilename, seed=1234+i, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

weights_path = "../out/retrained_regNN_oversampled_0_{}/it_{}.ckpt".format(index, i)
adam_model_001_1.save_weights(weights_path)