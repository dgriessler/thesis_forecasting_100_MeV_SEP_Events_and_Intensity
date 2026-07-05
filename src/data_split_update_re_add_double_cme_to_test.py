import numpy as np
import csv
import random
import math

def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def removeRows(data, feature, val):
    new_data = []
    for i in range(0, len(data)):
        elem = data[i]
        if feature in elem.keys():
            elem_val = elem[feature]
            if int(elem_val) != val:
                new_data.append(elem)
        else:
            new_data.append(elem)

    return new_data

def writeToCSV(data, filename):
    with open(filename, 'w', newline="") as csvfile:
        if len(data) > 0:
            fieldnames = []

            dataDict = data[0]
            for key in dataDict:
                fieldnames.append(key)

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
            writer.writeheader()
            for i in range(0, len(data)):
                writer.writerow(data[i])                

def normalizeFeatures(data, features):
    max_of_features = {}
    min_of_features = {}
    for feature in features:
        max_of_features[feature] = float(data[0][feature])
        min_of_features[feature] = float(data[0][feature])

    for elem in data:
        for feature in features:
            try:
                float_feature = float(elem[feature])
                if float_feature > max_of_features[feature]:
                    max_of_features[feature] = float_feature
                if float_feature < min_of_features[feature]:
                    min_of_features[feature] = float_feature
            except:
                print("IGNORING: {}. Feature: {}".format(elem, feature))

    log_before_normalize_features = []

    for feature in features:
        if feature == "diffusive_shock" or feature == "Type_2_Area":
            log_before_normalize_features.append(feature)
            print(feature)

    normalized_data = []
    for elem in data:
        new_elem = {}
        for feature in elem.keys():
            if feature in log_before_normalize_features:
                try:
                    float_feature = float(elem[feature])

                    if math.isclose(float_feature, 0):
                        new_elem[feature] = 0
                    elif math.isclose(min_of_features[feature], 0):
                        ln_float_feature = math.log(float_feature)
                        ln_max = math.log(max_of_features[feature])
                        new_elem[feature] = (ln_float_feature) / (ln_max)
                    else:
                        ln_float_feature = math.log(float_feature)
                        ln_max = math.log(max_of_features[feature])
                        ln_min = math.log(min_of_features[feature])
                        if math.isclose(ln_max - ln_min, 0):
                            new_elem[feature] = (ln_float_feature - ln_min)
                        else:
                            new_elem[feature] = (ln_float_feature - ln_min) / (ln_max - ln_min)
                except:
                    print("EXCEPTION: " + feature + " " + str(elem[feature]) + " " + str(min_of_features[feature]) + " " + str(max_of_features[feature]))
                    new_elem[feature] = elem[feature]
            else:
                try:
                    float_feature = float(elem[feature])
                    if math.isclose(max_of_features[feature] - min_of_features[feature], 0):
                        new_elem[feature] = (float_feature - min_of_features[feature])
                    else:
                        new_elem[feature] = (float_feature - min_of_features[feature]) / (max_of_features[feature] - min_of_features[feature])
                except:
                    new_elem[feature] = elem[feature]
        normalized_data.append(new_elem)

    return normalized_data

def mapByIndex(data):
    m = {}
    for elem in data:
        m[int(elem["index"])] = elem
    return m

def update_first_stage_training(data):
    features_to_normalize = [
        "donki_speed",
        "donki_ha",
        "longitude",
        "latitude",
        "Accel",
        "2nd_order_speed_final",
        "2nd_order_speed_20R",
        "Central_PA",
        "MPA",
        "sunspots",
        "halo",
        "Type_2_Area",
        "richardson_formula_degrees_phi_2_solar_wind",
        "diffusive_shock",
        "V log V",
        "CMEs_past_month",
        "CMEs_past_9_hours",
        "CMEs_over_1000_past_9_hrs",
        "Max_speed_past_day"
    ]

    double_cme_data = removeRows(data, feature="Double_CME_100_MeV", val=0)

    normalized_data = normalizeFeatures(double_cme_data, features_to_normalize)

    filenames = ["../res/gen/firstStageTest.csv", "../res/gen/syn/firstStageTestSyn.csv"]

    for filename in filenames:
        d = readCSVFile(filename)
        keys = d[0].keys()
        for elem in normalized_data:
            new_elem = dict()
            for key in keys:
                new_elem[key] = elem[key]
            d.append(new_elem)
        writeToCSV(d, filename)

    return normalized_data

def update_second_stage_training(normalized_data):
    for i in range(1, 10):
        testFilename = "../res/gen/secondStageOversampleTest_percentSEP_{:.1f}.csv".format((0.1 * i))

        filenames = [testFilename]

        for filename in filenames:
            d = readCSVFile(filename)
            keys = d[0].keys()
            for elem in normalized_data:
                new_elem = dict()
                for key in keys:
                    new_elem[key] = elem[key]
                d.append(new_elem)
            writeToCSV(d, filename)

def main():
    data_file = "../res/adapted_rRT_data_learn_richardson.csv"
    data = readCSVFile(data_file)

    normalized_data = update_first_stage_training(data)
    update_second_stage_training(normalized_data)

if __name__ == "__main__":
    main()
