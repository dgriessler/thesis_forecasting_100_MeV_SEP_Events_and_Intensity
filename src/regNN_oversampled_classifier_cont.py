import sys
from classifier import *

index = int(sys.argv[1])

cla = Classifier()

trainingData_x, trainingData_y  = cla.getFirstStageInput("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
validationData = cla.getValidationData("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))

baseStr = "../out/classifier/regNN_oversampled_0_{}/adam_model_001_01_oversampled_0_{}".format(index, index)

adam_model_001_01_checkpoint_path = baseStr + ".ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = baseStr + ".csv"
numEpochs = 70000

adam_model_001_1, adam_history_001_1 = cla.regNNValidation(adamLearningRate, trainingData_x, trainingData_y, validationData, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs, weights_filename=adam_model_001_01_checkpoint_path)
adam_model_001_1.save_weights(adam_model_001_01_checkpoint_path)
