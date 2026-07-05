import csv
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
from base_func import *
from keras.constraints import Constraint

def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def denseLossConvertIntensity(intensity):
    return int(intensity * math.pow(10, 5))

def LeakyReLU(alpha=0.2):
    return lambda x: tf.keras.backend.maximum(alpha * x, x)

seed = 1234
def getNextSeed():
    global seed
    next_seed = seed
    seed = seed + 1
    return next_seed

class FreezeSlice(Constraint):
    """
    Constraint which keeps a certain slice frozen at chosen values

    INPUTS:

    values - An object which can be converted into a numpy ndarray. These are
             the pre-chosen values. When using this constraint, the user should
             ensure that the dtype of values can be converted to the desired
             dtype of the weight tensor.

    slice - A slice or tuple of slices (it is recommended to use numpy.s_ to
            specify this parameter). This specifies which entries should be
            filled with the pre-chosen values. When using this initializer,
            the user should ensure that the slice object "fits inside" the shape
            of the tensor to be initialized, and that the resulting slice of the
            tensor has the same shape as the values ndarray.

    Source: https://github.com/keras-team/keras/issues/2880
    """
    def __init__(self, values, slice):
        if hasattr(values, "numpy"):
            self.values = values.numpy()
        elif isinstance(values, np.ndarray):
            self.values = values
        else:
            try:
                self.values = values.to_numpy()
            except:
                self.values = np.array(values)

        self.values = values
        self.slice = slice

    def __call__(self, w):
        zs = np.zeros(w.shape)
        zs[self.slice] = self.values
        os = np.ones(w.shape)
        os[self.slice] = 0
        return w * os + zs

def freeze_weights(model):
    """
    Freeze weights in model 
    """
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'trainable'):
            model.layers[ix].trainable = False

def get_feature_extractor(
    base_model: tf.keras.models.Model,
    freeze_model_weights: bool=True,
    lastHiddenLayerIndex: int=2,
    **kwargs
) -> tf.keras.models.Model:
    """
    Get feature extractor from base model and freeze weights

    :param base_model: Base training model
    :param freeze_weights: Flag to set to freeze feature extractor weights after
     we have extracted it from model.

    :return feature_extractor: Feature extractor layers of input model
    """
    feature_extractor = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[lastHiddenLayerIndex].output,
        name="feature_extractor"
    )

    # freeze feature extractor weights, if desired
    if freeze_model_weights:
        freeze_weights(feature_extractor)

    feature_extractor.summary()
    return feature_extractor

features_with_dense_loss = readCSVFile("../denseLoss/features_with_dense_loss_alt.csv")
dense_loss_alpha_0_0 = []
dense_loss_alpha_0_1 = []
dense_loss_alpha_0_2 = []
dense_loss_alpha_0_3 = []
dense_loss_alpha_0_4 = []
dense_loss_alpha_0_5 = []
dense_loss_alpha_0_6 = []
dense_loss_alpha_0_7 = []
dense_loss_alpha_0_8 = []
dense_loss_alpha_0_9 = []
dense_loss_alpha_1_0 = []
dense_loss_alpha_1_5 = []
dense_loss_alpha_2_0 = []
intensities = []
for elem in features_with_dense_loss:
    intensity = float(elem["100MeV_peak_intensity_ln"])
    converted_intensity = denseLossConvertIntensity(intensity)
    intensities.append(converted_intensity)

    dense_loss_alpha_0_0.append(float(elem["dense_loss_weighting_func_alpha_0.00"]))
    dense_loss_alpha_0_1.append(float(elem["dense_loss_weighting_func_alpha_0.10"]))
    dense_loss_alpha_0_2.append(float(elem["dense_loss_weighting_func_alpha_0.20"]))
    dense_loss_alpha_0_3.append(float(elem["dense_loss_weighting_func_alpha_0.30"]))
    dense_loss_alpha_0_4.append(float(elem["dense_loss_weighting_func_alpha_0.40"]))
    dense_loss_alpha_0_5.append(float(elem["dense_loss_weighting_func_alpha_0.50"]))
    dense_loss_alpha_0_6.append(float(elem["dense_loss_weighting_func_alpha_0.60"]))
    dense_loss_alpha_0_7.append(float(elem["dense_loss_weighting_func_alpha_0.70"]))
    dense_loss_alpha_0_8.append(float(elem["dense_loss_weighting_func_alpha_0.80"]))
    dense_loss_alpha_0_9.append(float(elem["dense_loss_weighting_func_alpha_0.90"]))
    dense_loss_alpha_1_0.append(float(elem["dense_loss_weighting_func_alpha_1.00"]))
    dense_loss_alpha_1_5.append(float(elem["dense_loss_weighting_func_alpha_1.50"]))
    dense_loss_alpha_2_0.append(float(elem["dense_loss_weighting_func_alpha_2.00"]))

dense_loss_keys_tensor = tf.constant(intensities)
dense_loss_values_0_0_tensor = tf.constant(dense_loss_alpha_0_0)
dense_loss_values_0_1_tensor = tf.constant(dense_loss_alpha_0_1)
dense_loss_values_0_2_tensor = tf.constant(dense_loss_alpha_0_2)
dense_loss_values_0_3_tensor = tf.constant(dense_loss_alpha_0_3)
dense_loss_values_0_4_tensor = tf.constant(dense_loss_alpha_0_4)
dense_loss_values_0_5_tensor = tf.constant(dense_loss_alpha_0_5)
dense_loss_values_0_6_tensor = tf.constant(dense_loss_alpha_0_6)
dense_loss_values_0_7_tensor = tf.constant(dense_loss_alpha_0_7)
dense_loss_values_0_8_tensor = tf.constant(dense_loss_alpha_0_8)
dense_loss_values_0_9_tensor = tf.constant(dense_loss_alpha_0_9)
dense_loss_values_1_0_tensor = tf.constant(dense_loss_alpha_1_0)
dense_loss_values_1_5_tensor = tf.constant(dense_loss_alpha_1_5)
dense_loss_values_2_0_tensor = tf.constant(dense_loss_alpha_2_0)

dense_loss_table_0_0 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_0_tensor),
    default_value=-100000)

dense_loss_table_0_1 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_1_tensor),
    default_value=-100000)

dense_loss_table_0_2 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_2_tensor),
    default_value=-100000)

dense_loss_table_0_3 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_3_tensor),
    default_value=-100000)

dense_loss_table_0_4 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_4_tensor),
    default_value=-100000)

dense_loss_table_0_5 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_5_tensor),
    default_value=-100000)

dense_loss_table_0_6 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_6_tensor),
    default_value=-100000)

dense_loss_table_0_7 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_7_tensor),
    default_value=-100000)

dense_loss_table_0_8 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_8_tensor),
    default_value=-100000)

dense_loss_table_0_9 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_9_tensor),
    default_value=-100000)

dense_loss_table_1_0 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_1_0_tensor),
    default_value=-100000)

dense_loss_table_1_5 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_1_5_tensor),
    default_value=-100000)

dense_loss_table_2_0 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_2_0_tensor),
    default_value=-100000)

def DenseLossWithMeanSquaredErrorAndAlpha0_0(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_0.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_1(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_1.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_2(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_2.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_3(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_3.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_4(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_4.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_5(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_5.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_6(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_6.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_7(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_7.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_8(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_8.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_9(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_9.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha1_0(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_1_0.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha1_5(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_1_5.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s

def DenseLossWithMeanSquaredErrorAndAlpha2_0(y_true, y_pred):
    s = 0.0
    w_sum = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_2_0.lookup(denseLossConvertIntensity(y_true[i][0]))
        w_sum = w_sum + lookup
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / w_sum
    return s
