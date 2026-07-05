from extra_features import *

def calculate_trained_richardson_ln(data, trained_richardson_ln_feature_name):
    # Richardson's LN eq: LN(I) = LN(w_exp) + w_v * V - connection_angle^2 / (2 * 43^2)
    richardson_coefficients = readCSVFile("../res/trained/richardson.csv")[0]
    ln_w_exp = math.log(float(richardson_coefficients["w_exp"]))
    w_v = float(richardson_coefficients["w_v"])
    for elem in data:
        V = float(elem["donki_speed"])
        connection_angle_degrees = float(elem["connection_angle_degrees"])
        trained_ln_richardson = ln_w_exp + w_v * V - ((connection_angle_degrees * connection_angle_degrees) / (2 * 43 * 43))
        elem[trained_richardson_ln_feature_name] = trained_ln_richardson

def main():
    data = readCSVFile("../res/adapted_rRT_data_learn_richardson.csv")
    fieldnames = list(data[0].keys())

    trained_richardson_ln_feature_name = "trained_richardson_ln"
    calculate_trained_richardson_ln(data, trained_richardson_ln_feature_name)
    fieldnames.append(trained_richardson_ln_feature_name)

    # Resort to the provided order (by index)
    data.sort(key=sortDataIndex)
    writeCSVFile(data, "../res/adapted_rRT_data_learn_richardson.csv", fieldnames)

if __name__ == "__main__":
    main()