import sys
from classifier import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

cla = Classifier()

all_trainingData_x, all_trainingData_y  = cla.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

adam_model_001_1_retrained_checkpoint_path = "../out/classifier/regNN_retrained/adam_model_001_1_retrained.ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/classifier/regNN_retrained/adam_model_001_1_retrained.csv"

weights_filename = adam_model_001_1_retrained_checkpoint_path
adam_model_001_1_retrained, adam_history_001_1_retrained = cla.regNN(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, 1234, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)

# NN 5 iterations for data

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y = cla.getFirstStageInput("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))

numIterations = 5

adamLearningRate = 0.001
alpha = 0.3

for i in range(0, numIterations):
    # restore model
    adam_model_001_1_retrained.load_weights(adam_model_001_1_retrained_checkpoint_path)

    logFilename = "../out/classifier/retrained_rRT_0_{}/it_{}.csv".format(index, i)
    
    NNmodel, NNModelHistory = cla.rRT(adamLearningRate, adam_model_001_1_retrained, all_trainingData_x, all_trainingData_y, logFilename, 1234+i, adamEpsilon, alpha=alpha, numEpochs=numEpochs)

    NNmodel_weights_path = "../out/classifier/retrained_rRT_0_{}/it_{}.ckpt".format(index, i)
    NNmodel.save_weights(NNmodel_weights_path)
