from regression_alt import *
import sys

denseLossAlpha = float(sys.argv[1])
numEpochs = int(sys.argv[2])
i = int(sys.argv[3])

reg = RegressionAlt()

all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/alt_denseLoss_retrained/alpha_{:.2f}_it_{}.csv".format(denseLossAlpha, i)

adam_model_001_1, adam_history_001_1 = reg.denseLossNNModel(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, seed=1234+i, denseLossAlpha=denseLossAlpha, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)
adam_model_001_1_checkpoint_path = "../out/alt_denseLoss_retrained/alpha_{:.2f}_it_{}.ckpt".format(denseLossAlpha, i)
adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)
