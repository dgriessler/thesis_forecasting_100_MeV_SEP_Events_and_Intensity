import sys
from regression_trained_richardson import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionTrainedRichardson()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/trained/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
validationData = reg.getValidationData("../res/trained/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))

baseStr = "../out/using_trained_richardson/regNN_oversampled_0_{}/adam_model_001_01_oversampled_0_{}".format(index, index)

adam_model_001_01_checkpoint_path = baseStr + ".ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = baseStr + ".csv"

adam_model_001_1, adam_history_001_1 = reg.regNNValidation(adamLearningRate, trainingData_x, trainingData_y, validationData, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs, weights_filename=adam_model_001_01_checkpoint_path)

adam_model_001_1, adam_history_001_1 = reg.regNNValidationCont(adam_model_001_1, trainingData_x, trainingData_y, validationData, logFilename, numEpochs=numEpochs)

adam_model_001_1.save_weights(adam_model_001_01_checkpoint_path)
