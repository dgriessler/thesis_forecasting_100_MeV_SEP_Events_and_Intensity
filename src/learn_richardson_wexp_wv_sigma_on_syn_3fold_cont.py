from regression import *
import sys

group = int(sys.argv[1])
subset = int(sys.argv[2])

reg = Regression()

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardsonLearnWExpWVSigma("../res/gen/fold/syn/firstStageTraining_3Fold_Syn_{}_{}.csv".format(group, subset))
richardson_validationData = reg.getValidationDataRichardsonLearnWExpWVSigma("../res/gen/fold/syn/firstStageValidation_3Fold_Syn_{}_{}.csv".format(group, subset))

adamLearningRate = 0.0001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/learn_richardson_wexp_wv_sigma_on_syn_3fold/group_{}_subset_{}.csv".format(group, subset)
numEpochs = 220000

adam_model_001_1_checkpoint_path = "../out/learn_richardson_wexp_wv_sigma_on_syn_3fold/group_{}_subset_{}.ckpt".format(group, subset)
adam_model_001_1, adam_history_001_1 = reg.trainRichardsonWExpWVSigmaValidation(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, richardson_validationData, logFilename, adamEpsilon=adamEpsilon, numEpochs=numEpochs, weights_filename=adam_model_001_1_checkpoint_path)

adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)



