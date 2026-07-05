from regression_richardson_mixed import *

reg = RegressionRichardsonMixed()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = reg.getValidationData("../res/gen/firstStageValidation.csv")

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

numEpochs = 50000

autoencoder_trainingData_x = trainingData_x
autoencoder_trainingData_y = trainingData_y
autoencoder_validationData = validationData

logFilename = "../out/richardson_mixed/autoencoder_finding_alpha/autoencoder_finding_alpha_autoencoder.csv"
autoencoder_validation_weights_path = "../out/richardson_mixed/autoencoder_finding_alpha/autoencoder_finding_alpha_autoencoder.ckpt"

autoencoder_model = reg.richardsonMixed_autoencoder_finding_alpha_auto(adamOptimizer, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilename, numEpochs, autoencoder_validation_weights_path, alpha)

keras.utils.plot_model(autoencoder_model, to_file='../out/richardson_mixed/autoencoder_finding_alpha/autoencoder_finding_alpha_model.png', show_shapes=True, show_layer_names=True)
