import sys
from regression_richardson_mixed import *

i = int(sys.argv[1])

reg = RegressionRichardsonMixed()

all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/firstStageAllTraining.csv", False)

combined_training_data_x = (richardson_trainingData_x, all_trainingData_x)
combined_training_data_y = all_trainingData_y

adam_model_001_01_retrained_checkpoint_path = "../out/richardson_mixed/regNN_retrained/adam_model_001_1_retrained.ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/richardson_mixed/regNN_retrained/it_{}.csv".format(i)

numEpochs = 39069

adam_model_001_1_retrained, adam_history_001_1_retrained = reg.richardsonMixedNN(adamLearningRate, combined_training_data_x, combined_training_data_y, logFilename, seed=1234+i, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)
adam_model_001_01_retrained_checkpoint_path = "../out/richardson_mixed/regNN_retrained/it_{}.ckpt".format(i)
adam_model_001_1_retrained.save_weights(adam_model_001_01_retrained_checkpoint_path)


