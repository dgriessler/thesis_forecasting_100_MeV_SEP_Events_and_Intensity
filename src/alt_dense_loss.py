from regression_alt import *
import sys

denseLossAlpha = float(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionAlt()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = reg.getValidationData("../res/gen/firstStageValidation.csv")

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/alt_denseLoss/alpha_{:.2f}.csv".format(denseLossAlpha)

adam_model_001_1, adam_history_001_1 = reg.denseLossNNModelValidation(adamLearningRate, trainingData_x, trainingData_y, validationData, logFilename, denseLossAlpha=denseLossAlpha, adamEpsilon=adamEpsilon, alpha=alpha, doTrain=True, numEpochs=numEpochs)
adam_model_001_1_checkpoint_path = "../out/alt_denseLoss/alpha_{:.2f}.ckpt".format(denseLossAlpha)
adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)
