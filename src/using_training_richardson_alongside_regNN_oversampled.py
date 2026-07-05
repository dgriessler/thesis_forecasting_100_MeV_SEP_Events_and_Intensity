import sys
from regression_using_training_richardson import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionUsingTrainingRichardson()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
validationData = reg.getValidationData("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index), False)
richardson_validationData = reg.getValidationDataRichardson("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index), False)

combined_training_data_x = (richardson_trainingData_x, trainingData_x)
combined_training_data_y = trainingData_y
combined_validation_data = ((richardson_validationData[0], validationData[0]), richardson_validationData[1])

baseStr = "../out/using_training_richardson_alongside/regNN_oversampled_0_{}/adam_model_001_01_oversampled_0_{}".format(index, index)

adam_model_001_01_checkpoint_path = baseStr + ".ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = baseStr + ".csv"

adam_model_001_1, adam_history_001_1 = reg.learnedRichardsonAlongsideRegularNNValidation(adamLearningRate, combined_training_data_x, combined_training_data_y, combined_validation_data, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)
adam_model_001_1.save_weights(adam_model_001_01_checkpoint_path)
