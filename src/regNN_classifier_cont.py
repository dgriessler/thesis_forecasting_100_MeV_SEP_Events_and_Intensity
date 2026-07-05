from classifier import *

cla = Classifier()

trainingData_x, trainingData_y  = cla.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = cla.getValidationData("../res/gen/firstStageValidation.csv")

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = cla.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/classifier/regNN/adam_model_001_1.csv"
numEpochs = 50000

adam_model_001_1_checkpoint_path = "../out/classifier/regNN/adam_model_001_1.ckpt"
adam_model_001_1, adam_history_001_1 = cla.regNNValidation(adamLearningRate, trainingData_x, trainingData_y, validationData, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs, weights_filename=adam_model_001_1_checkpoint_path)

adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)
