import numpy as np
import csv
import random
import math
import os

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

def splitSEPEvents(data):
    eventsSEP = []
    eventsElevated = []
    eventsBackground = []
    for row in data:
        try:
            target_val = int(row["target"])
            if target_val == 1:
                eventsSEP.append(row)
            elif target_val == 2:
                eventsElevated.append(row)
            else:
                eventsBackground.append(row)
        except:
            print("Invalid row: {}".format(row))
    return (eventsSEP, eventsElevated, eventsBackground)

def stratifyTrainingValidationTestSEPElevatedWith3Fold(sepEvents, elevatedEvents):
    numTrainingExamples = 3
    numValidationExamples = 1
    numTestExamples = 2
    numBuckets = 6
    
    sortedEvents = sorted(sepEvents + elevatedEvents, key = lambda x: x["100MeV_peak_intensity"], reverse=True)
    
    trainingSets = [[list(), list(), list(), list()], [list(), list(), list(), list()], [list(), list(), list(), list()]]
    validationSets = [[list(), list(), list(), list()], [list(), list(), list(), list()], [list(), list(), list(), list()]]
    testSets = [list(), list(), list()]
        
    bucketSize = numTrainingExamples + numValidationExamples + numTestExamples
    for i in range(0, len(sortedEvents), bucketSize):
        nextBucketEvents = sortedEvents[i : i + bucketSize]
        
        randomSelection = random.sample(nextBucketEvents, len(nextBucketEvents))
        
        if len(nextBucketEvents) == 6:
            trainingValidationCombined = [list(), list(), list()]
            trainingValidationCombined[0] = randomSelection[2:6]
            testSets[0].extend(randomSelection[0:2])

            trainingValidationCombined[1] = randomSelection[0:2] + randomSelection[4:6]
            testSets[1].extend(randomSelection[2:4])

            trainingValidationCombined[2] = randomSelection[0:4]
            testSets[2].extend(randomSelection[4:6])
            
            for j in range(0, 3):
                trainingSets[j][0].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][1], trainingValidationCombined[j][2]])
                validationSets[j][0].extend([trainingValidationCombined[j][3]])

                trainingSets[j][1].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][2], trainingValidationCombined[j][3]])
                validationSets[j][1].extend([trainingValidationCombined[j][1]])

                trainingSets[j][2].extend([trainingValidationCombined[j][1], trainingValidationCombined[j][2], trainingValidationCombined[j][3]])
                validationSets[j][2].extend([trainingValidationCombined[j][0]])

                trainingSets[j][3].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][1], trainingValidationCombined[j][3]])
                validationSets[j][3].extend([trainingValidationCombined[j][2]])
            
        elif len(nextBucketEvents) == 5:
            trainingValidationCombined = [list(), list(), list()]
            trainingValidationCombined[0] = randomSelection[2:5]
            testSets[0].extend(randomSelection[0:2])

            trainingValidationCombined[1] = randomSelection[0:2] + randomSelection[4:5]
            testSets[1].extend(randomSelection[2:4])

            trainingValidationCombined[2] = randomSelection[0:4]
            testSets[2].extend(randomSelection[4:5])
            
            for j in range(0, 2):
                trainingSets[j][0].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][1]])
                validationSets[j][0].extend([trainingValidationCombined[j][2]])

                trainingSets[j][1].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][2]])
                validationSets[j][1].extend([trainingValidationCombined[j][1]])

                trainingSets[j][2].extend([trainingValidationCombined[j][1], trainingValidationCombined[j][2]])
                validationSets[j][2].extend([trainingValidationCombined[j][0]])

                #trainingSets[j][3].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][1]])
                #validationSets[j][3].extend([trainingValidationCombined[j][2]])
            
            j = 2
            trainingSets[j][0].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][1], trainingValidationCombined[j][2]])
            validationSets[j][0].extend([trainingValidationCombined[j][3]])

            trainingSets[j][1].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][2], trainingValidationCombined[j][3]])
            validationSets[j][1].extend([trainingValidationCombined[j][1]])

            trainingSets[j][2].extend([trainingValidationCombined[j][1], trainingValidationCombined[j][2], trainingValidationCombined[j][3]])
            validationSets[j][2].extend([trainingValidationCombined[j][0]])

            trainingSets[j][3].extend([trainingValidationCombined[j][0], trainingValidationCombined[j][1], trainingValidationCombined[j][3]])
            validationSets[j][3].extend([trainingValidationCombined[j][2]])
    
    trainingSEPSets = [[list(), list(), list(), list()], [list(), list(), list(), list()], [list(), list(), list(), list()]]
    trainingElevatedSets = [[list(), list(), list(), list()], [list(), list(), list(), list()], [list(), list(), list(), list()]]
    
    for i in range(0, 3):
        for j in range(0, 4):
            for elem in trainingSets[i][j]:
                if int(elem["target"]) == 1:
                    trainingSEPSets[i][j].append(elem)
                elif int(elem["target"]) == 2:
                    trainingElevatedSets[i][j].append(elem)
                else:
                    print("Unrecognized target: {}".format(elem["target"]))
    
    validationSEPSets = [[list(), list(), list(), list()], [list(), list(), list(), list()], [list(), list(), list(), list()]]
    validationElevatedSets = [[list(), list(), list(), list()], [list(), list(), list(), list()], [list(), list(), list(), list()]]
    
    for i in range(0, 3):
        for j in range(0, 4):
            for elem in validationSets[i][j]:
                if int(elem["target"]) == 1:
                    validationSEPSets[i][j].append(elem)
                elif int(elem["target"]) == 2:
                    validationElevatedSets[i][j].append(elem)
                else:
                    print("Unrecognized target: {}".format(elem["target"]))
    
    testSEPSets = [list(), list(), list()]
    testElevatedSets = [list(), list(), list()]
    
    for i in range(0, 3):
        for elem in testSets[i]:
            if int(elem["target"]) == 1:
                testSEPSets[i].append(elem)
            elif int(elem["target"]) == 2:
                testElevatedSets[i].append(elem)
            else:
                print("Unrecognized target: {}".format(elem["target"]))
    
    return (trainingSEPSets, validationSEPSets, testSEPSets, trainingElevatedSets, validationElevatedSets, testElevatedSets)

def create_stratified(sepEvents, elevatedEvents, trainingBackground, validationBackground, testBackground):
    trainingSEPs, validationSEPs, testSEPs, trainingElevateds, validationElevateds, testElevateds = stratifyTrainingValidationTestSEPElevatedWith3Fold(sepEvents, elevatedEvents)

    folder = "../res/gen/fold"
    syn_folder = "../res/gen/fold/syn"

    for k in range(0, 3):
        for j in range(0, 4):
            writeToCSV(trainingSEPs[k][j] + trainingElevateds[k][j] + trainingBackground, folder + "/" + "firstStageTraining_3Fold_{}_{}.csv".format(k, j))
            writeToCSV(validationSEPs[k][j] + validationElevateds[k][j] + validationBackground, folder + "/" + "firstStageValidation_3Fold_{}_{}.csv".format(k, j))
        
            writeToCSV(trainingSEPs[k][j] + trainingElevateds[k][j], syn_folder + "/" + "firstStageTraining_3Fold_Syn_{}_{}.csv".format(k, j))
            writeToCSV(validationSEPs[k][j] + validationElevateds[k][j], syn_folder + "/" + "firstStageValidation_3Fold_Syn_{}_{}.csv".format(k, j))

        writeToCSV(trainingSEPs[k][0] + validationSEPs[k][0] + trainingElevateds[k][0] + validationElevateds[k][0] + trainingBackground + validationBackground, folder + "/" + "firstStageAllTraining_3Fold_{}.csv".format(k))
        writeToCSV(trainingSEPs[k][0] + validationSEPs[k][0] + trainingElevateds[k][0] + validationElevateds[k][0], syn_folder + "/" + "firstStageAllTraining_Syn_3Fold_{}.csv".format(k))

        writeToCSV(trainingSEPs[k][0] + trainingElevateds[k][0] + trainingBackground, folder + "/" + "firstStageTraining_3Fold_{}.csv".format(k))
        writeToCSV(validationSEPs[k][0] + validationElevateds[k][0] + validationBackground, folder + "/" + "firstStageValidation_3Fold_{}.csv".format(k))

    for j in range(0, 3):
        writeToCSV(testSEPs[j] + testElevateds[j] + testBackground, folder + "/" + "firstStageTest_3Fold_{}.csv".format(j))
        writeToCSV(testSEPs[j] + testElevateds[j], syn_folder + "/" + "firstStageTest_3Fold_Syn_{}.csv".format(j))

def mapByIndex(data):
    m = {}
    for elem in data:
        m[int(elem["index"])] = elem
    return m

def remove_which(data, feature_name, value):
    new_data = []
    for elem in data:
        if elem[feature_name] != value:
            new_data.append(elem)
    return new_data

def replaceData(replace_data, replace_with_by_index):
    for elem in replace_data:
        new_elem = replace_with_by_index[int(elem["index"])]
        for key in elem.keys():
            elem[key] = new_elem[key]

def create_syn(data):
    index_map_data = mapByIndex(data)

    folder = "../res/gen/fold"
    syn_folder = "../res/gen/fold/syn"

    for k in range(0, 3):
        for j in range(0, 4):
            training_filename = folder + "/" + "firstStageTraining_3Fold_{}_{}.csv".format(k, j)
            training_data = readCSVFile(training_filename)
            filtered_sep_elevated_training = remove_which(training_data, "target", 0)
            replaceData(filtered_sep_elevated_training, index_map_data)
            writeToCSV(filtered_sep_elevated_training, syn_folder + "/" + "firstStageTraining_3Fold_Syn_{}_{}.csv".format(k, j))

            validation_filename = folder + "/" + "firstStageValidation_3Fold_{}_{}.csv".format(k, j)
            validation_data = readCSVFile(validation_filename)
            filtered_sep_elevated_validation = remove_which(validation_data, "target", 0)
            replaceData(filtered_sep_elevated_validation, index_map_data)
            writeToCSV(filtered_sep_elevated_validation, syn_folder + "/" + "firstStageValidation_3Fold_Syn_{}_{}.csv".format(k, j))          

        all_training_filename = folder + "/" + "firstStageAllTraining_3Fold_{}_{}.csv".format(k, j)
        all_training_data = readCSVFile(all_training_filename)
        filtered_sep_elevated_all_training = remove_which(all_training_data, "target", 0)
        replaceData(filtered_sep_elevated_all_training, index_map_data)
        writeToCSV(filtered_sep_elevated_all_training, syn_folder + "/" + "firstStageAllTraining_Syn_3Fold_{}.csv".format(k))

    for j in range(0, 3):
        test_filename = folder + "/" + "firstStageTest_3Fold_{}.csv".format(j)
        test_data = readCSVFile(test_filename)
        filtered_sep_elevated_test = remove_which(test_data, "target", 0)
        replaceData(filtered_sep_elevated_test, index_map_data)
        writeToCSV(filtered_sep_elevated_test, syn_folder + "/" + "firstStageTest_3Fold_Syn_{}.csv".format(j))

def main():
    data_file = "../res/adapted_rRT_data_learn_richardson.csv"
    data = readCSVFile(data_file)

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

    normalized_data = normalizeFeatures(data, features_to_normalize)

    eventsSEP, eventsElevated, eventsBackground = splitSEPEvents(normalized_data)

    # Total
    # 16 Elevated
    # 13 SEP
    # 2256 Background

    # Training
    # 8 Elevated
    # 7 SEP
    # 1240 Background

    # Validation
    # 4 Elevated
    # 3 SEP
    # 340 Background

    # Test
    # 4 Elevated
    # 3 SEP
    # 676 Background
    randomBackgroundSelection = random.sample(eventsBackground, len(eventsBackground))
    background_num_training = 1240
    background_num_validation = 340
    background_num_test = 676
    background_training = randomBackgroundSelection[0:background_num_training]
    background_validation = randomBackgroundSelection[background_num_training : background_num_training + background_num_validation]
    background_test = randomBackgroundSelection[background_num_training + background_num_validation : ]
    
    create_stratified(eventsSEP, eventsElevated, background_training, background_validation, background_test)

    create_syn(data)

    try:
        os.remove("../res/gen/firstStageTraining.csv")
    except:
        pass
    try:
        os.remove("../res/gen/firstStageValidation.csv")
    except:
        pass
    try:
        os.remove("../res/gen/firstStageTest.csv")
    except:
        pass

if __name__ == "__main__":
    main()
