from regression import *

reg = Regression()

richardson_trainingData_x, richardson_trainingData_y  = reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageAllTrainingSyn.csv")

adamLearningRate = 0.0001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init_long.csv"
numEpochs = 148510

adam_model_001_1, adam_history_001_1 = reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, numEpochs=numEpochs)
adam_model_001_1_checkpoint_path = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init_long.ckpt"
adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)
