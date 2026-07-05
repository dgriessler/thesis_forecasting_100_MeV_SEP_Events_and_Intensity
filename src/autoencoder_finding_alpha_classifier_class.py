from classifier import *

cla = Classifier()

trainingData_x, trainingData_y  = cla.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = cla.getValidationData("../res/gen/firstStageValidation.csv")


# Retrain with training + validation examples combined up to the epoch where validation error is minimized
all_trainingData_x, all_trainingData_y  = cla.getFirstStageInput("../res/gen/firstStageAllTraining.csv")

adamLearningRate = 0.0001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizerClassifier = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

numEpochs = 50000

autoencoder_trainingData_x = trainingData_x
autoencoder_trainingData_y = trainingData_y
autoencoder_validationData = validationData

logFilenameClassifier = "../out/classifier/autoencoder_finding_alpha/autoencoder_finding_alpha_classifier.csv"
autoencoder_validation_weights_path_classifier = "../out/classifier/autoencoder_finding_alpha/autoencoder_finding_alpha_classifier.ckpt"

classifier_model = cla.autoencoder_finding_alpha_class(adamOptimizerClassifier, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilenameClassifier, numEpochs, alpha, autoencoder_validation_weights_path_classifier)

keras.utils.plot_model(classifier_model, to_file='../out/classifier/autoencoder_finding_alpha/classifier_finding_alpha_model.png', show_shapes=True, show_layer_names=True)
