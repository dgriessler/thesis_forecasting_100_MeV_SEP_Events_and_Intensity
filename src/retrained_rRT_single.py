import sys
from regression import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])
i = int(sys.argv[3])

reg = Regression()

all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

adam_model_001_1_retrained_checkpoint_path = "../out/regNN_retrained/adam_model_001_1_retrained.ckpt"

retrained_adamLearningRate = 0.001
retrained_adamEpsilon = 1.0
retrained_alpha = 0.3
retrained_logFilename = "../out/regNN_retrained/adam_model_001_1_retrained.csv"

retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path

# NN 5 iterations for data

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x_0_1, all_trainingData_y_0_1 = reg.getFirstStageInput("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))
testData_x_0_1, testData_y_0_1 = reg.getFirstStageInput("../res/gen/secondStageOversampleTest_percentSEP_0.{}.csv".format(index))
testData_target_0_1 = reg.getTargets("../res/gen/secondStageOversampleTest_percentSEP_0.{}.csv".format(index))

numIterations = 5

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

adam_model_001_1_retrained, adam_history_001_1_retrained = reg.regNN(retrained_adamLearningRate, all_trainingData_x, all_trainingData_y, retrained_logFilename, seed=1234, adamEpsilon=retrained_adamEpsilon, alpha=retrained_alpha, weights_filename=retrained_weights_filename)

feature_extractor = get_feature_extractor(adam_model_001_1_retrained)

logFilename = "../out/retrained_rRT_0_{}/it_{}.csv".format(index, i)
    
NNmodel, NNModelHistory = reg.rRT(adamLearningRate, feature_extractor, all_trainingData_x_0_1, all_trainingData_y_0_1, logFilename, seed=1234+i, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

NNmodel_weights_path = "../out/retrained_rRT_0_{}/it_{}.ckpt".format(index, i)
NNmodel.save_weights(NNmodel_weights_path)
