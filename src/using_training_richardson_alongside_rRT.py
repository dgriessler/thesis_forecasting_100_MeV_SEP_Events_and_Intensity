from regression_using_training_richardson import *

index = int(sys.argv[1])
numEpochs = int(sys.argv[2])

reg = RegressionUsingTrainingRichardson()

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3

trained_checkpoint_path = "../out/using_training_richardson_alongside/regNN/adam_model_001_1.ckpt"
adam_model_001_1, _ = reg.learnedRichardsonAlongsideRegularNN(adamLearningRate, None, None, None, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=trained_checkpoint_path, doTrain=False)

feature_extractor = get_feature_extractor(adam_model_001_1, lastHiddenLayerIndex=len(adam_model_001_1.layers) - 2)

# Second stage use class based sampling to retrain regressive weights.
trainingData_x, trainingData_y = reg.getFirstStageInput("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index))
validationData = reg.getValidationData("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index))

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/secondStageOversampleTraining_percentSEP_0.{}.csv".format(index), False)
richardson_validationData = reg.getValidationDataRichardson("../res/gen/secondStageOversampleValidation_percentSEP_0.{}.csv".format(index), False)

combined_training_data_x = (richardson_trainingData_x, trainingData_x)
combined_training_data_y = trainingData_y
combined_validation_data = ((richardson_validationData[0], validationData[0]), richardson_validationData[1])

logFilename = "../out/using_training_richardson_alongside/rRT_0_{}/rRT_model_0_{}.csv".format(index, index)

rRT_model, _ = reg.learnedRichardsonAlongsideRRTValidation(adamLearningRate, feature_extractor, combined_training_data_x, combined_training_data_y, combined_validation_data, logFilename, adamEpsilon=adamEpsilon, alpha=alpha, seed=1234+index, numEpochs=numEpochs)

rRT_model_weights_path = "../out/using_training_richardson_alongside/rRT_0_{}/rRT_model_0_{}.ckpt".format(index, index)
rRT_model.save_weights(rRT_model_weights_path)
