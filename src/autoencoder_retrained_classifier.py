from classifier import *

cla = Classifier()

# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = cla.getFirstStageInput("../res/gen/firstStageAllTraining.csv")


adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

numEpochs = 20788

autoencoder_trainingData_x = all_trainingData_x
autoencoder_trainingData_y = [all_trainingData_x, all_trainingData_y]

logFilename = "../out/classifier/autoencoder_retrained/autoencoder_validation_training_retrained.csv"

autoencoder_model = cla.autoencoderAll(adamOptimizer, logFilename, autoencoder_trainingData_x, autoencoder_trainingData_y, numEpochs, alpha)
keras.utils.plot_model(autoencoder_model, to_file='../out/classifier/autoencoder_retrained/autoencoder_model.png', show_shapes=True, show_layer_names=True)

autoencoder_validation_weights_path = "../out/classifier/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
autoencoder_model.save_weights(autoencoder_validation_weights_path)
