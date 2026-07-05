import sys
from regression_trained_richardson import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionTrainedRichardson()

all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/trained/firstStageAllTraining.csv")

adam_model_001_1_retrained_checkpoint_path = "../out/using_trained_richardson/regNN_retrained/adam_model_001_1_retrained.ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/using_trained_richardson/regNN_retrained/adam_model_001_1_retrained.csv"

weights_filename = adam_model_001_1_retrained_checkpoint_path
adam_model_001_1_retrained, adam_history_001_1_retrained = reg.regNN(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, seed=1234, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)

feature_extractor = get_feature_extractor(adam_model_001_1_retrained)

# Second stage use class based sampling to retrain regressive weights.
adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/using_trained_richardson/rRT_0_{}/rRT_model_0_{}.csv".format(index, index)

secondStageTrainingData_x_oversample, secondStageTrainingData_y_oversample = reg.getFirstStageInput("../res/trained/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
secondStageValidationData_oversample = reg.getValidationData("../res/trained/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))

rRT_model, rRT_history = reg.rRTValidation(adamLearningRate, feature_extractor, secondStageTrainingData_x_oversample, secondStageTrainingData_y_oversample, secondStageValidationData_oversample, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

rRT_model_weights_path = "../out/using_trained_richardson/rRT_0_{}/rRT_model_0_{}.ckpt".format(index, index)
rRT_model.save_weights(rRT_model_weights_path)