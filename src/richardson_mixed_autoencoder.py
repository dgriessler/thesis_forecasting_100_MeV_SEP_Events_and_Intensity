from regression_richardson_mixed import *

reg = RegressionRichardsonMixed()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = reg.getValidationData("../res/gen/firstStageValidation.csv")

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/firstStageTraining.csv", False)
richardson_validationData = reg.getValidationDataRichardson("../res/gen/firstStageValidation.csv", False)


adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

numEpochs = 30000

autoencoder_trainingData_x = (richardson_trainingData_x, trainingData_x)
autoencoder_trainingData_y = [trainingData_x, trainingData_y]
autoencoder_validationData = ([richardson_validationData[0], validationData[0]], [validationData[0], richardson_validationData[1]])

logFilename = "../out/richardson_mixed/autoencoder/autoencoder_validation_training.csv"

autoencoder_model = reg.richardsonMixed_autoencoder(adamOptimizer, logFilename, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, numEpochs, alpha)
keras.utils.plot_model(autoencoder_model, to_file='../out/richardson_mixed/autoencoder/autoencoder_model.png', show_shapes=True, show_layer_names=True)

autoencoder_validation_weights_path = "../out/richardson_mixed/autoencoder/autoencoder_validation_training.ckpt"
autoencoder_model.save_weights(autoencoder_validation_weights_path)
