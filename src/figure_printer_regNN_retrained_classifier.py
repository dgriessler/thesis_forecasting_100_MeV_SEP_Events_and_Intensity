from regression import *

def create_learned_richardson_layers():
    w_v = float(1.0)
    w_exp = float(1.0)

    # Input layer Richardson
    inputRichardsonLayer = keras.Input(shape=(2,), name="input_richardson")

    # Output Layer Richardson
    # I = 0.013 * exp(0.0036 * V - phi^2 / (2 * sigma^2))
    # LN(I) = LN(0.013) + 0.0036 * V + 1 * (- phi^2 / (2 * sigma^2))
    # input params: 1) speed 2) connection_angle bundle
    # So, connection_angle bundle weight should be 1.0 fixed.
    # We learned w_v (the 0.0036 term in richardson) and w_exp (the 0.013 term in richardson)
    # The actual term there is LN(w_exp)
    weights = [np.empty(shape = (2,1), dtype = np.float32), np.empty(shape = (1), dtype = np.float32)]
    weights[0][0][0] = w_v
    weights[0][1][0] = 1.0
    weights[1][0] = math.log(w_exp)
    outputRichardsonLayer = keras.layers.Dense(1, weights=weights, trainable=False, activation=None, name="output_richardson")(inputRichardsonLayer)

    return (inputRichardsonLayer, outputRichardsonLayer)

adamLearningRate = 0.0001
adamEpsilon = 1.0
alpha = 0.3
adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

inputRichardsonLayer, outputRichardsonLayer = create_learned_richardson_layers()


initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=1234)
# Input layer regular NN
inputLayerReg = keras.Input(shape=(18,))

# Hidden layer regular NN
encoderLayerReg = keras.layers.Dense(16, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(inputLayerReg)

zLayerReg = keras.layers.Dense(9, kernel_initializer=initializer, activation=LeakyReLU(alpha=alpha))(encoderLayerReg)

# Output Layer regular NN
outputLayerReg = layers.Dense(1, activation=None, kernel_initializer=initializer)(zLayerReg)

combined = keras.layers.concatenate([outputRichardsonLayer, outputLayerReg], name="concatenate")

# Mixing output from regular NN and output from richardson
combinedOutputLayer = keras.layers.Dense(1, kernel_initializer=initializer, activation=None)(combined)

   
model = keras.Model(inputs=[inputRichardsonLayer, inputLayerReg], outputs=combinedOutputLayer, name="adapted_rRT_" + str(adamLearningRate))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.MeanSquaredError())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon), loss=keras.losses.MeanSquaredError())

keras.utils.plot_model(model, to_file='../eval/re_network_train.png', show_shapes=True, show_layer_names=True)
