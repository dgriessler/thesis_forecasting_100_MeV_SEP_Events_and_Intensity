from regression_trained_richardson import *

reg = RegressionTrainedRichardson()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/trained/firstStageTraining.csv")
validationData = reg.getValidationData("../res/trained/firstStageValidation.csv")


# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = reg.getFirstStageInput("../res/trained/firstStageAllTraining.csv")


adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

numEpochs = 12186

autoencoder_trainingData_x = all_trainingData_x
autoencoder_trainingData_y = [all_trainingData_x, all_trainingData_y]

logFilename = "../out/using_trained_richardson/autoencoder_retrained/autoencoder_validation_training_retrained.csv"

autoencoder_model = reg.autoencoderAll(adamOptimizer, logFilename, autoencoder_trainingData_x, autoencoder_trainingData_y, numEpochs, alpha)
keras.utils.plot_model(autoencoder_model, to_file='../out/using_trained_richardson/autoencoder_retrained/autoencoder_model.png', show_shapes=True, show_layer_names=True)

autoencoder_validation_weights_path = "../out/using_trained_richardson/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
autoencoder_model.save_weights(autoencoder_validation_weights_path)
