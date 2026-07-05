from regression import *

reg = Regression()

richardson_trainingData_x, richardson_trainingData_y = reg.getFirstStageInputRichardson("../res/firstStageTrainingSyn.csv", useSynData=True)

adamLearningRate = 0.0001
adamEpsilon = 1.0
alpha = 0.3
logFilename = "../out/learn_richardson_syn/adam_model_001_1_reg_init.csv"
numEpochs = 10000

adam_model_001_1_checkpoint_path = "../out/learn_richardson_syn/adam_model_001_1_reg_init.ckpt"
adam_model_001_1, adam_history_001_1 = reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, numEpochs=numEpochs, weights_filename=adam_model_001_1_checkpoint_path)

layerNum = 0
for layer in adam_model_001_1.layers:
    weights = layer.get_weights() # list of numpy arrays
    print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
    layerNum = layerNum + 1
