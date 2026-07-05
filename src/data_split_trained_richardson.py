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
    mappedData = {}
    for elem in data:
        mappedData[elem["index"]] = elem
    return mappedData

def applyFeatureToData(dataDestination, mappedDataSource, featureName):
    for elem in dataDestination:
        elem_source = mappedDataSource[elem["index"]]
        elem[featureName] = elem_source[featureName]

def create_first_stage_training(data):
    trained_richardson_ln_feature_name = "trained_richardson_ln"

    features_to_normalize = [
        trained_richardson_ln_feature_name
    ]

    normalized_data = normalizeFeatures(data, features_to_normalize)
    mapped_normalized_data = mapByIndex(normalized_data)

    firstStageAllTraining = readCSVFile("../res/gen/firstStageAllTraining.csv")
    applyFeatureToData(firstStageAllTraining, mapped_normalized_data, trained_richardson_ln_feature_name)

    firstStageTraining = readCSVFile("../res/gen/firstStageTraining.csv")
    applyFeatureToData(firstStageTraining, mapped_normalized_data, trained_richardson_ln_feature_name)

    firstStageValidation = readCSVFile("../res/gen/firstStageValidation.csv")
    applyFeatureToData(firstStageValidation, mapped_normalized_data, trained_richardson_ln_feature_name)

    firstStageTest = readCSVFile("../res/gen/firstStageTest.csv")
    applyFeatureToData(firstStageTest, mapped_normalized_data, trained_richardson_ln_feature_name)

    writeToCSV(firstStageTraining, "../res/trained/firstStageTraining.csv")
    writeToCSV(firstStageValidation, "../res/trained/firstStageValidation.csv")
    writeToCSV(firstStageAllTraining, "../res/trained/firstStageAllTraining.csv")
    writeToCSV(firstStageTest, "../res/trained/firstStageTest.csv")

def create_second_stage_training(data):
    trained_richardson_ln_feature_name = "trained_richardson_ln"

    features_to_normalize = [
        trained_richardson_ln_feature_name
    ]

    normalized_data = normalizeFeatures(data, features_to_normalize)
    mapped_normalized_data = mapByIndex(normalized_data)

    for i in range(1, 10):
        sourceTrainingFilename = "../res/gen/secondStageOversampleTraining_percentSEP_{:.1f}.csv".format((0.1 * i))
        sourceTraining = readCSVFile(sourceTrainingFilename)
        applyFeatureToData(sourceTraining, mapped_normalized_data, trained_richardson_ln_feature_name)

        trainingFilename = "../res/trained/secondStageOversampleTraining_percentSEP_{:.1f}.csv".format((0.1 * i))
        print("WRITING TRAINING: {}".format(trainingFilename))
        writeToCSV(sourceTraining, trainingFilename)

        sourceValidationFilename = "../res/gen/secondStageOversampleValidation_percentSEP_{:.1f}.csv".format((0.1 * i))
        sourceValidation = readCSVFile(sourceValidationFilename)
        applyFeatureToData(sourceValidation, mapped_normalized_data, trained_richardson_ln_feature_name)

        validationFilename = "../res/trained/secondStageOversampleValidation_percentSEP_{:.1f}.csv".format((0.1 * i))
        print("WRITING VALIDATION: {}".format(validationFilename))
        writeToCSV(sourceValidation, validationFilename)

        sourceAllTrainingFilename = "../res/gen/secondStageOversampleAllTraining_percentSEP_{:.1f}.csv".format((0.1 * i))
        sourceAllTraining = readCSVFile(sourceAllTrainingFilename)
        applyFeatureToData(sourceAllTraining, mapped_normalized_data, trained_richardson_ln_feature_name)

        allTrainingFilename = "../res/trained/secondStageOversampleAllTraining_percentSEP_{:.1f}.csv".format((0.1 * i))
        print("WRITING ALL TRAINING: {}".format(allTrainingFilename))
        writeToCSV(sourceAllTraining, allTrainingFilename)

        sourceTestFilename = "../res/gen/secondStageOversampleTest_percentSEP_{:.1f}.csv".format((0.1 * i))
        sourceTest = readCSVFile(sourceTestFilename)
        applyFeatureToData(sourceTest, mapped_normalized_data, trained_richardson_ln_feature_name)

        testFilename = "../res/trained/secondStageOversampleTest_percentSEP_{:.1f}.csv".format((0.1 * i))
        print("WRITING TEST: {}".format(testFilename))
        writeToCSV(sourceTest, testFilename)

def main():
    data_file = "../res/adapted_rRT_data_learn_richardson.csv"
    data = readCSVFile(data_file)

    create_first_stage_training(data)
    create_second_stage_training(data)

if __name__ == "__main__":
    main()
