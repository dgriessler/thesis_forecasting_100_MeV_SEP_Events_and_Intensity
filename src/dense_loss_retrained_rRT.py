import sys
from regression import *

denseLossAlpha = float(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = Regression()

all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

for i in range(0, 5):
    adam_model_001_1_retrained_checkpoint_path = "../out/regNN_retrained/adam_model_001_1_retrained.ckpt"

    adamLearningRate = 0.001
    adamEpsilon = 1.0
    alpha = 0.3

    weights_filename = adam_model_001_1_retrained_checkpoint_path
    adam_model_001_1_retrained, adam_history_001_1_retrained = reg.regNN(adamLearningRate, None, None, None, 1234, adamEpsilon, alpha, weights_filename=weights_filename)

    feature_extractor = get_feature_extractor(adam_model_001_1_retrained)

    # Second stage use class based sampling to retrain regressive weights.
    adamLearningRate = 0.001
    adamEpsilon = 1.0
    alpha = 0.3
    logFilename = "../out/denseLoss_retrained_rRT/alpha_{:.2f}_it_{}.csv".format(denseLossAlpha, i)

    rRT_model, rRT_history = reg.denseLoss_rRT(adamLearningRate, feature_extractor, all_trainingData_x, all_trainingData_y, logFilename, denseLossAlpha, 1234+i, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

    rRT_model_weights_path = "../out/denseLoss_retrained_rRT/alpha_{:.2f}_it_{}.ckpt".format(denseLossAlpha, i)
    rRT_model.save_weights(rRT_model_weights_path)