from regression import *
import sys

group = int(sys.argv[1])
numEpochs = int(sys.argv[2])
learningRate = float(sys.argv[3])

reg = Regression()

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardsonLearnWVSigma("../res/gen/fold/syn/firstStageAllTraining_Syn_3Fold_{}.csv".format(group))

w_0 = reg.getFixedW0Richardson("../res/gen/fold/syn/firstStageAllTraining_Syn_3Fold_{}.csv".format(group))

adamLearningRate = learningRate
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/retrained_learn_richardson_wv_sigma_on_syn_3fold/group_{}_lR_{:.5f}.csv".format(group, adamLearningRate)

adam_model_001_1, adam_history_001_1 = reg.trainRichardsonLearnWVSigma(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, w_0=w_0, numEpochs=numEpochs)
adam_model_001_1_checkpoint_path = "../out/retrained_learn_richardson_wv_sigma_on_syn_3fold/group_{}_lR_{:.5f}.ckpt".format(group, adamLearningRate)
adam_model_001_1.save_weights(adam_model_001_1_checkpoint_path)



