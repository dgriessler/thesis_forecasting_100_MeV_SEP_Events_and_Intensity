from regression_trained_richardson import *

reg = RegressionTrainedRichardson()

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/trained/firstStageAllTraining.csv")


adam_model_001_01_retrained_checkpoint_path = "../out/using_trained_richardson/regNN_retrained/adam_model_001_1_retrained.ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
numEpochs = 11872
alpha = 0.3
logFilename = "../out/using_trained_richardson/regNN_retrained/adam_model_001_1_retrained.csv"

adam_model_001_1_retrained, adam_history_001_1_retrained = reg.regNN(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, seed=1234, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)
adam_model_001_01_retrained_checkpoint_path = "../out/using_trained_richardson/regNN_retrained/adam_model_001_1_retrained.ckpt"
adam_model_001_1_retrained.save_weights(adam_model_001_01_retrained_checkpoint_path)


