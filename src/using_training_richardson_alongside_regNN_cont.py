import sys
from regression import *

numEpochs = int(sys.argv[1])

reg = Regression()

trainingData_x, trainingData_y = reg.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = reg.getValidationData("../res/gen/firstStageValidation.csv")

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/firstStageTraining.csv", False)
richardson_validationData = reg.getValidationDataRichardson("../res/gen/firstStageValidation.csv", False)

combined_training_data_x = (richardson_trainingData_x, trainingData_x)
combined_training_data_y = trainingData_y
combined_validation_data = ((richardson_validationData[0], validationData[0]), richardson_validationData[1])

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

logFilename = "../out/using_training_richardson_alongside/regNN/adam_model_001_1.csv"

adam_model_001_1_checkpoint_path = "../out/using_training_richardson_alongside/regNN/adam_model_001_1.ckpt"
adam_model_001_1, _ = reg.learnedRichardsonAlongsideRegularNNValidation(adamLearningRate, combined_training_data_x, combined_training_data_y, combined_validation_data, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs, weights_filename=adam_model_001_1_checkpoint_path, doTrain=True)

adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)
