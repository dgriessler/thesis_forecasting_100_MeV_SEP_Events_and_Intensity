from regression import *

reg = Regression()

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/syn/firstStageTrainingSyn.csv", useSynData=True)
richardson_validationData = reg.getValidationDataRichardson("../res/gen/syn/firstStageValidationSyn.csv", useSynData=True)

adamLearningRate = 0.0001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/learn_richardson_syn/adam_model_001_1_reg_init.csv"
numEpochs = 100000

adam_model_001_1_checkpoint_path = "../out/learn_richardson_syn/adam_model_001_1_reg_init.ckpt"
adam_model_001_1, adam_history_001_1 = reg.trainRichardsonValidation(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, richardson_validationData, logFilename, adamEpsilon=adamEpsilon, numEpochs=numEpochs, weights_filename=adam_model_001_1_checkpoint_path)
adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)



