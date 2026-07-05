import sys
from classifier import *

index = int(sys.argv[1])

cla = Classifier()

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = cla.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

adam_model_001_1_retrained_checkpoint_path = "../out/classifier/regNN_retrained/adam_model_001_1_retrained.ckpt"

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/classifier/regNN_retrained/adam_model_001_1_retrained.csv"

weights_filename = adam_model_001_1_retrained_checkpoint_path
adam_model_001_1_retrained, adam_history_001_1_retrained = cla.regNN(adamLearningRate, all_trainingData_x, all_trainingData_y, logFilename, 1234, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)

# restore model
adam_model_001_1_retrained.load_weights(adam_model_001_1_retrained_checkpoint_path)

# Second stage use class based sampling to retrain regressive weights.
adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/classifier/rRT_0_{}/rRT_model_0_{}.csv".format(index, index)
numEpochs = 15000

secondStageTrainingData_x, secondStageTrainingData_y = cla.getFirstStageInput("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
secondStageValidationData = cla.getValidationData("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))
rRT_model, rRT_history = cla.rRTValidation(adamLearningRate, adam_model_001_1_retrained, secondStageTrainingData_x, secondStageTrainingData_y, secondStageValidationData, logFilename, adamEpsilon, alpha=alpha, numEpochs=numEpochs)

rRT_model_weights_path = "../out/classifier/rRT_0_{}/rRT_model_0_{}.ckpt".format(index, index)
rRT_model.save_weights(rRT_model_weights_path)
