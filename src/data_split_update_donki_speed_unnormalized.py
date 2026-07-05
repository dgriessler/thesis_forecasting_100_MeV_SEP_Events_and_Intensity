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
    index_map_data = mapByIndex(data)

    filenames = ["../res/gen/firstStageTraining.csv", "../res/gen/firstStageValidation.csv", "../res/gen/firstStageAllTraining.csv", "../res/gen/firstStageTest.csv", "../res/gen/syn/firstStageTrainingSyn.csv", "../res/gen/syn/firstStageValidationSyn.csv", "../res/gen/syn/firstStageAllTrainingSyn.csv", "../res/gen/syn/firstStageTestSyn.csv"]

    for filename in filenames:
        d = readCSVFile(filename)
        for elem in d:
            new_elem = index_map_data[int(elem["index"])]
            if new_elem is not None:
                elem["donki_speed_unnormalized"] = new_elem["donki_speed"]
            else:
                print("ERROR: INDEX NOT FOUND {}".format(int(elem["index"])))
        writeToCSV(d, filename)

    return index_map_data

def update_second_stage_training(index_map_data):
    for i in range(1, 10):
        trainingFilename = "../res/gen/secondStageOversampleTraining_percentSEP_{:.1f}.csv".format((0.1 * i))
        validationFilename = "../res/gen/secondStageOversampleValidation_percentSEP_{:.1f}.csv".format((0.1 * i))
        allTrainingFilename = "../res/gen/secondStageOversampleAllTraining_percentSEP_{:.1f}.csv".format((0.1 * i))
        testFilename = "../res/gen/secondStageOversampleTest_percentSEP_{:.1f}.csv".format((0.1 * i))

        filenames = [trainingFilename, validationFilename, allTrainingFilename, testFilename]

        for filename in filenames:
            d = readCSVFile(filename)
            for elem in d:
                new_elem = index_map_data[int(elem["index"])]
                if new_elem is not None:
                    elem["donki_speed_unnormalized"] = new_elem["donki_speed"]
                else:
                    print("ERROR: INDEX NOT FOUND {}".format(int(elem["index"])))
            writeToCSV(d, filename)


def update_stratified_richardson(data):
    folder = "../res/gen/fold"
    syn_folder = "../res/gen/fold/syn"

    index_map_data = mapByIndex(data)

    for k in range(0, 3):
        for j in range(0, 4):
            filenames = [
                syn_folder + "/" + "firstStageTraining_3Fold_Syn_{}_{}.csv".format(k, j),
                syn_folder + "/" + "firstStageValidation_3Fold_Syn_{}_{}.csv".format(k, j)
            ]
            for filename in filenames:
                d = readCSVFile(filename)
                for elem in d:
                    new_elem = index_map_data[int(elem["index"])]
                    if new_elem is not None:
                        elem["donki_speed_unnormalized"] = new_elem["donki_speed"]
                    else:
                        print("ERROR: INDEX NOT FOUND {}".format(int(elem["index"])))
                writeToCSV(d, filename)


        filenames = [
            syn_folder + "/" + "firstStageAllTraining_Syn_3Fold_{}.csv".format(k)
        ]
        for filename in filenames:
            d = readCSVFile(filename)
            for elem in d:
                new_elem = index_map_data[int(elem["index"])]
                if new_elem is not None:
                    elem["donki_speed_unnormalized"] = new_elem["donki_speed"]
                else:
                    print("ERROR: INDEX NOT FOUND {}".format(int(elem["index"])))
            writeToCSV(d, filename)

    for j in range(0, 3):
        filenames = [
            syn_folder + "/" + "firstStageTest_3Fold_Syn_{}.csv".format(j)
        ]
        for filename in filenames:
            d = readCSVFile(filename)
            for elem in d:
                new_elem = index_map_data[int(elem["index"])]
                if new_elem is not None:
                    elem["donki_speed_unnormalized"] = new_elem["donki_speed"]
                else:
                    print("ERROR: INDEX NOT FOUND {}".format(int(elem["index"])))
            writeToCSV(d, filename)

    for k in range(0, 3):
        for j in range(0, 4):
            filenames = [
                folder + "/" + "firstStageTraining_3Fold_{}_{}.csv".format(k, j),
                folder + "/" + "firstStageValidation_3Fold_{}_{}.csv".format(k, j),
            ]
            for filename in filenames:
                d = readCSVFile(filename)
                for elem in d:
                    new_elem = index_map_data[int(elem["index"])]
                    if new_elem is not None:
                        elem["donki_speed_unnormalized"] = new_elem["donki_speed"]
                    else:
                        print("ERROR: INDEX NOT FOUND {}".format(int(elem["index"])))
                writeToCSV(d, filename)

        filenames = [
            folder + "/" + "firstStageAllTraining_3Fold_{}.csv".format(k),
        ]
        for filename in filenames:
            d = readCSVFile(filename)
            for elem in d:
                new_elem = index_map_data[int(elem["index"])]
                if new_elem is not None:
                    elem["donki_speed_unnormalized"] = new_elem["donki_speed"]
                else:
                    print("ERROR: INDEX NOT FOUND {}".format(int(elem["index"])))
            writeToCSV(d, filename)

    for j in range(0, 3):
        filenames = [
            folder + "/" + "firstStageTest_3Fold_{}.csv".format(j),
        ]
        for filename in filenames:
            d = readCSVFile(filename)
            for elem in d:
                new_elem = index_map_data[int(elem["index"])]
                if new_elem is not None:
                    elem["donki_speed_unnormalized"] = new_elem["donki_speed"]
                else:
                    print("ERROR: INDEX NOT FOUND {}".format(int(elem["index"])))
            writeToCSV(d, filename)

def main():
    data_file = "../res/adapted_rRT_data_learn_richardson.csv"
    data = readCSVFile(data_file)

    mapped_data = update_first_stage_training(data)
    update_second_stage_training(mapped_data)

    update_stratified_richardson(data)

if __name__ == "__main__":
    main()
