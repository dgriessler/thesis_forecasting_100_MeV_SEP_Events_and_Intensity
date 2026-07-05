from regression_using_training_richardson import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionUsingTrainingRichardson()

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

# Second stage use class based sampling to retrain regressive weights.
trainingData_x, trainingData_y = reg.getFirstStageInput("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index))

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/secondStageOversampleAllTraining_percentSEP_0.{}.csv".format(index), False)

combined_training_data_x = (richardson_trainingData_x, trainingData_x)
combined_training_data_y = trainingData_y

numIterations = 5

for i in range(0, numIterations):
    trained_checkpoint_path = "../out/using_training_richardson_alongside/regNN/adam_model_001_1.ckpt"
    adam_model_001_1, _ = reg.learnedRichardsonAlongsideRegularNN(adamLearningRate, None, None, None, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=trained_checkpoint_path, doTrain=False)

    feature_extractor = get_feature_extractor(adam_model_001_1, lastHiddenLayerIndex=len(adam_model_001_1.layers) - 2)

    logFilename = "../out/using_training_richardson_alongside/retrained_rRT_0_{}/it_{}.csv".format(index, i)
    rRT_model, _ = reg.learnedRichardsonAlongsideRRT(adamLearningRate, feature_extractor, combined_training_data_x, combined_training_data_y, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, seed=1234+i, numEpochs=numEpochs)

    rRT_model_weights_path = "../out/using_training_richardson_alongside/retrained_rRT_0_{}/it_{}.ckpt".format(index, i)
    rRT_model.save_weights(rRT_model_weights_path)
