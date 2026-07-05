from regression import *
import sys

group = int(sys.argv[1])
subset = int(sys.argv[2])

reg = Regression()

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/fold/syn/no_double_cme_firstStageTraining_3Fold_Syn_{}_{}.csv".format(group, subset))
richardson_validationData = reg.getFirstStageInputRichardson("../res/gen/fold/syn/no_double_cme_firstStageValidation_3Fold_Syn_{}_{}.csv".format(group, subset))

adamLearningRate = 0.0001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/learn_richardson_on_syn_3fold_no_double_cme/group_{}_subset_{}.csv".format(group, subset)
numEpochs = 100000

adam_model_001_1_checkpoint_path = "../out/learn_richardson_on_syn_3fold_no_double_cme/group_{}_subset_{}.ckpt".format(group, subset)
adam_model_001_1, adam_history_001_1 = reg.trainRichardsonValidation(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, richardson_validationData, logFilename, adamEpsilon=adamEpsilon, numEpochs=numEpochs, weights_filename=adam_model_001_1_checkpoint_path)
adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)



