import sys
from classifier import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

cla = Classifier()

all_trainingData_x, all_trainingData_y = cla.getFirstStageInput("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

numIterations = 5

for i in range(0, numIterations):
  logFilename = "../out/classifier/retrained_regNN_oversampled_0_{}/it_{}.csv".format(index, i)

  adam_model_001_1, adam_history_001_1 = cla.regNN(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, 1234+i, adamEpsilon=adamEpsilon, alpha=alpha, numEpochs=numEpochs)

  weights_path = "../out/classifier/retrained_regNN_oversampled_0_{}/it_{}.ckpt".format(index, i)
  adam_model_001_1.save_weights(weights_path)
