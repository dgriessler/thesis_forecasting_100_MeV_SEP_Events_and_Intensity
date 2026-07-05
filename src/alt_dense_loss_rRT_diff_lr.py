import sys
from regression_alt import *

denseLossAlpha = float(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionAlt()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = reg.getValidationData("../res/gen/firstStageValidation.csv")

adam_model_001_1_retrained_checkpoint_path = "../out/regNN_retrained/adam_model_001_1_retrained.ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

weights_filename = adam_model_001_1_retrained_checkpoint_path
adam_model_001_1_retrained, adam_history_001_1_retrained = reg.regNN(adamLearningRate, None, None, None, 1234, adamEpsilon, alpha, weights_filename=weights_filename)

feature_extractor = get_feature_extractor(adam_model_001_1_retrained)

# Second stage use class based sampling to retrain regressive weights.
adamLearningRate = 0.0001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/alt_denseLoss_rRT/lr_{:.5f}_alpha_{:.2f}.csv".format(adamLearningRate, denseLossAlpha)

rRT_model, rRT_history = reg.denseLoss_rRTValidation(adamLearningRate, feature_extractor, trainingData_x, trainingData_y, validationData, logFilename, denseLossAlpha, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

rRT_model_weights_path = "../out/alt_denseLoss_rRT/lr_{:.5f}_alpha_{:.2f}.ckpt".format(adamLearningRate, denseLossAlpha)
rRT_model.save_weights(rRT_model_weights_path)