from regression_richardson_mixed import *

reg = RegressionRichardsonMixed()

trainingData_x, trainingData_y = reg.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/firstStageAllTraining.csv", False)

combined_training_data_x = (richardson_trainingData_x, trainingData_x)
combined_training_data_y = trainingData_y

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

logFilename = "../out/richardson_mixed/regNN_retrained/adam_model_001_1.csv"
numEpochs = 39069

adam_model_001_1, _ = reg.richardsonMixedNN(adamLearningRate, combined_training_data_x, combined_training_data_y, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)
adam_model_001_1_checkpoint_path = "../out/richardson_mixed/regNN_retrained/adam_model_001_1.ckpt"
adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)
