from regression_richardson_mixed import *

reg = RegressionRichardsonMixed()

trainingData_x, trainingData_y  = reg.getFirstStageInput("../res/gen/firstStageTraining.csv")
validationData = reg.getValidationData("../res/gen/firstStageValidation.csv")


richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/gen/firstStageTraining.csv", False)
richardson_validationData = reg.getValidationDataRichardson("../res/gen/firstStageValidation.csv", False)

combined_training_data_x = (richardson_trainingData_x, trainingData_x)
combined_training_data_y = trainingData_y
combined_validation_data = ((richardson_validationData[0], validationData[0]), richardson_validationData[1])

adamLearningRate = 0.001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizerClassifier = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

numEpochs = 50000

autoencoder_trainingData_x = combined_training_data_x
autoencoder_trainingData_y = combined_training_data_y
autoencoder_validationData = combined_validation_data

logFilenameClassifier = "../out/richardson_mixed/autoencoder_finding_alpha/autoencoder_finding_alpha_classifier.csv"

autoencoder_validation_weights_path_classifier = "../out/richardson_mixed/autoencoder_finding_alpha/autoencoder_finding_alpha_classifier.ckpt"

classifier_model = reg.richardsonMixed_autoencoder_finding_alpha_class(adamOptimizerClassifier, autoencoder_trainingData_x, autoencoder_trainingData_y, autoencoder_validationData, logFilenameClassifier, numEpochs, alpha, autoencoder_validation_weights_path_classifier)

keras.utils.plot_model(classifier_model, to_file='../out/richardson_mixed/autoencoder_finding_alpha/classifier_finding_alpha_model.png', show_shapes=True, show_layer_names=True)
