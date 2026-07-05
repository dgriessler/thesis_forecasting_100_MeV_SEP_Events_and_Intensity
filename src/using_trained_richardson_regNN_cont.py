from regression_trained_richardson import *

reg = RegressionTrainedRichardson()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/trained/firstStageTraining.csv")
validationData = reg.getValidationData("../res/trained/firstStageValidation.csv")

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/trained/firstStageAllTraining.csv")

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/using_trained_richardson/regNN/adam_model_001_1.csv"
numEpochs = 25000

adam_model_001_1_checkpoint_path = "../out/using_trained_richardson/regNN/adam_model_001_1.ckpt"
adam_model_001_1, adam_history_001_1 = reg.regNNValidation(adamLearningRate, trainingData_x, trainingData_y, validationData, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs, weights_filename=adam_model_001_1_checkpoint_path)

adam_model_001_1, adam_history_001_1 = reg.regNNValidationCont(adam_model_001_1, trainingData_x, trainingData_y, validationData, logFilename, numEpochs=numEpochs)

adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)
