import sys
from regression import *

index = int(sys.argv[1])

reg = Regression()

all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

adam_model_001_1_retrained_checkpoint_path = "../out/regNN_retrained/adam_model_001_1_retrained.ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/regNN_retrained/adam_model_001_1_retrained.csv"

weights_filename = adam_model_001_1_retrained_checkpoint_path
adam_model_001_1_retrained, adam_history_001_1_retrained = reg.regNN(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, seed=1234, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)

feature_extractor = get_feature_extractor(adam_model_001_1_retrained)

# Second stage use class based sampling to retrain regressive weights.
adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/rRT_0_{}/rRT_model_0_{}.csv".format(index, index)
numEpochs = 15000

secondStageTrainingData_x_oversample, secondStageTrainingData_y_oversample = reg.getFirstStageInput("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
secondStageValidationData_oversample = reg.getValidationData("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))

rRT_model, rRT_history = reg.rRTValidation(adamLearningRate, feature_extractor, secondStageTrainingData_x_oversample, secondStageTrainingData_y_oversample, secondStageValidationData_oversample, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

rRT_model_weights_path = "../out/rRT_0_{}/rRT_model_0_{}.ckpt".format(index, index)
rRT_model.save_weights(rRT_model_weights_path)