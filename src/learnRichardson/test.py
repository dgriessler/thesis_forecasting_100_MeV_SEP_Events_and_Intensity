import sys
from base_func import *

features_with_dense_loss = readCSVFile("../denseLoss/features_with_dense_loss.csv")
dense_loss_alpha_0_0 = []
dense_loss_alpha_0_5 = []
dense_loss_alpha_1_0 = []
dense_loss_alpha_1_5 = []
dense_loss_alpha_2_0 = []
intensities = []
for elem in features_with_dense_loss:
    intensity = float(elem["100MeV_peak_intensity_ln"])
    converted_intensity = int(intensity * math.pow(10, 6))
    intensities.append(converted_intensity)

    dense_loss_alpha_0_0.append(float(elem["dense_loss_weighting_func_alpha_0.00"]))
    dense_loss_alpha_0_5.append(float(elem["dense_loss_weighting_func_alpha_0.50"]))
    dense_loss_alpha_1_0.append(float(elem["dense_loss_weighting_func_alpha_1.00"]))
    dense_loss_alpha_1_5.append(float(elem["dense_loss_weighting_func_alpha_1.50"]))
    dense_loss_alpha_2_0.append(float(elem["dense_loss_weighting_func_alpha_2.00"]))

dense_loss_keys_tensor = tf.constant(intensities)
dense_loss_values_0_0_tensor = tf.constant(dense_loss_alpha_0_0)
dense_loss_values_0_5_tensor = tf.constant(dense_loss_alpha_0_5)
dense_loss_values_1_0_tensor = tf.constant(dense_loss_alpha_1_0)
dense_loss_values_1_5_tensor = tf.constant(dense_loss_alpha_1_5)
dense_loss_values_2_0_tensor = tf.constant(dense_loss_alpha_2_0)

dense_loss_table_0_0 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_0_tensor),
    default_value=1)

dense_loss_table_0_5 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_0_5_tensor),
    default_value=-1.0)

dense_loss_table_1_0 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_1_0_tensor),
    default_value=-1.0)

dense_loss_table_1_5 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_1_5_tensor),
    default_value=-1.0)

dense_loss_table_2_0 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dense_loss_keys_tensor, dense_loss_values_2_0_tensor),
    default_value=-1.0)

def DenseLossWithMeanSquaredErrorAndAlpha0_0(y_true, y_pred):
    N = len(y_true)
    if N == 0:
        N = 1.0
    else:
        N = float(N)
    s = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_0.lookup(tf.strings.as_string(y_pred[i][0]))
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / N
    return s

def DenseLossWithMeanSquaredErrorAndAlpha0_5(y_true, y_pred):
    N = len(y_true)
    if N == 0:
        N = 1.0
    else:
        N = float(N)
    s = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_0_5.lookup(tf.strings.as_string(y_pred[i][0]))
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / N
    return s

def DenseLossWithMeanSquaredErrorAndAlpha1_0(y_true, y_pred):
    N = len(y_true)
    if N == 0:
        N = 1.0
    else:
        N = float(N)
    s = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_1_0.lookup(tf.strings.as_string(y_pred[i][0]))
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / N
    return s

def DenseLossWithMeanSquaredErrorAndAlpha1_5(y_true, y_pred):
    N = len(y_true)
    if N == 0:
        N = 1.0
    else:
        N = float(N)
    s = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_1_5.lookup(tf.strings.as_string(y_pred[i][0]))
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / N
    return s

def DenseLossWithMeanSquaredErrorAndAlpha2_0(y_true, y_pred):
    N = len(y_true)
    if N == 0:
        N = 1.0
    else:
        N = float(N)
    s = 0.0
    for i in range(0, len(y_true)):
        lookup = dense_loss_table_2_0.lookup(tf.strings.as_string(y_pred[i][0]))
        s = s + lookup * (y_true[i][0] - y_pred[i][0]) * (y_true[i][0] - y_pred[i][0])
    s = s / N
    return s

print("\n\n")
print(intensities[58])

print('\n\n')
print(tf.constant(intensities[58]))

print("\n\n")
print(dense_loss_table_0_5.export())

print("\n\n")
print(dense_loss_table_0_5.lookup(tf.constant(intensities[58])))

#print("\n\n")
#print(dense_loss_table_0_5.lookup(tf.strings.as_string(tf.constant(float(intensities[59])))))

#print("\n\n")
#print(dense_loss_table_0_5.lookup(tf.strings.as_string(tf.constant(float(intensities[60])))))

#print("\n\n")
#print(dense_loss_table_0_5.lookup(tf.strings.as_string(tf.constant(float(intensities[61])))))

