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

def splitTrainingValidationTestExactly(events, numTrainingExamples, numValidationExamples, numTestExamples):
    
    print("Splitting {} Events into {}/{}/{} training/validation/test".format(len(events), numTrainingExamples, numValidationExamples, numTestExamples))
    
    # randomly select all events then split them up
    randomSelection = random.sample(events, len(events))

    startIndex = 0
    endIndex = numTrainingExamples
    trainingSet = randomSelection[startIndex:endIndex]

    startIndex = endIndex
    endIndex = endIndex + numValidationExamples
    validationSet = randomSelection[startIndex:endIndex]

    startIndex = endIndex
    testSet = randomSelection[startIndex:]

    return (trainingSet, validationSet, testSet)

def splitTrainingValidationTest(events, trainingPercent, validationPercent, testPercent, secondLevelBool):
    totalPercent = trainingPercent + validationPercent + testPercent
    if (not totalPercent.is_integer()) or (int(totalPercent) != 1):
        print("Percentages don't add to 1. Training: {}, Validation: {}, Test: {}".format(trainingPercent, validationPercent, testPercent))
        return
    
    print("Splitting {} events by {:.0f}/{:.0f}/{:.0f} training/validation/test".format(len(events), trainingPercent*100, validationPercent*100, testPercent*100))
    
    numTrainingExamples = int(trainingPercent * len(events))
    numTestExamples = int(testPercent * len(events))
    numValidationExamples = len(events) - numTrainingExamples - numTestExamples
    if validationPercent == 0:
        numTestExamples += numValidationExamples
        numValidationExamples = 0
    
    if numTestExamples == 0 and numValidationExamples > 0:
        numTestExamples = 1
        numValidationExamples = numValidationExamples - 1
       
    if secondLevelBool is None:
        trainingSet, validationSet, testSet = splitTrainingValidationTestExactly(events, numTrainingExamples, numValidationExamples, numTestExamples)

        return (trainingSet, validationSet, testSet)
    else:
        # Second level indicates that splitting should ensure that this bool feature is represented at the given percentages
        secondLevelEvents = []
        otherEvents = []
        for event in events:
            if int(event[secondLevelBool]) == 1:
                secondLevelEvents.append(event)
            else:
                otherEvents.append(event)
        
        print("Splitting second level \"{}\": {} t / {} f".format(secondLevelBool, len(secondLevelEvents), len(otherEvents)))
        
        secondLevelTrainingSet, secondLevelValidationSet, secondLevelTestSet = splitTrainingValidationTest(secondLevelEvents, trainingPercent=trainingPercent, validationPercent=validationPercent, testPercent=testPercent, secondLevelBool=None)
        remainingNumTrainingExamples = numTrainingExamples - len(secondLevelTrainingSet)
        remainingNumValidationExamples = numValidationExamples - len(secondLevelValidationSet)
        remainingNumTestExamples = numTestExamples - len(secondLevelTestSet)
        otherEventsTrainingSet, otherEventsValidationSet, otherEventsTestSet = splitTrainingValidationTestExactly(otherEvents, remainingNumTrainingExamples, remainingNumValidationExamples, remainingNumTestExamples)
        
        return (secondLevelTrainingSet + otherEventsTrainingSet, secondLevelValidationSet + otherEventsValidationSet, secondLevelTestSet + otherEventsTestSet)

def duplicate(l, totalLen):
    numCopies = int(totalLen / len(l))
    selection = []
    for i in range(0, numCopies):
        selection.extend(l)
    
    # randomly pick remaining to match the required number
    numRemaining = totalLen - len(l) * numCopies
    if numRemaining > 0:
        randomSelection = random.sample(l, numRemaining)
        selection.extend(randomSelection)

    return selection

def classbasedOversample(trainingSEPSet, validationSEPSet, testSEPSet,
                         trainingElevatedSet, validationElevatedSet, testElevatedSet,
                         trainingBackgroundSet, validationBackgroundSet, testBackgroundSet,
                         percentSEP, percentElevated):
    
    totalClassPercent = percentSEP + percentElevated
    if totalClassPercent < 0.0 or totalClassPercent > 1.0:
        print("Percent SEP + Percent Elevated is expected to be between 0 and 1: {}".format(totalClassPercent))
        return
    
    print("Replicating {} SEP events, {} Elevated events, and {} Background events via Class based sampling using {:.0f}% SEP events {:.0f}% Elevated events".format(len(trainingSEPSet) + len(validationSEPSet) + len(testSEPSet), len(trainingElevatedSet) + len(validationElevatedSet) + len(testElevatedSet), len(trainingBackgroundSet) + len(validationBackgroundSet) + len(testBackgroundSet), percentSEP*100, percentElevated*100))
     
    # Number of oversampled SEP events = Percentage of SEP * Number of background events / Percentage of background
    numTrainingEventsBackground = len(trainingBackgroundSet)
    numTrainingEventsSEP = int(percentSEP * numTrainingEventsBackground / (1 - totalClassPercent))
    numTrainingEventsElevated = int(percentElevated * numTrainingEventsBackground / (1 - totalClassPercent))
    
    print("Num Training events SEP: {}. Num Training events Elevated: {}. Num Training events Background {}.".format(numTrainingEventsSEP, numTrainingEventsElevated, numTrainingEventsBackground))
    
    # duplicate SEP training events
    selectionTrainingSEP = duplicate(trainingSEPSet, numTrainingEventsSEP)
    
    # duplicate Elevated training events
    selectionTrainingElevated = duplicate(trainingElevatedSet, numTrainingEventsElevated)

    # Validation and training are also combined for training
    # Generate a version where the training/validation percentages are combined and then oversampled
    allTrainingSEPSet = trainingSEPSet + validationSEPSet
    allTrainingElevatedSet = trainingElevatedSet + validationElevatedSet
    allTrainingBackgroundSet = trainingBackgroundSet + validationBackgroundSet
    # Number of oversampled SEP events = Percentage of SEP * Number of background events / Percentage of background
    numAllTrainingEventsBackground = len(allTrainingBackgroundSet)
    numAllTrainingEventsSEP = int(percentSEP * numAllTrainingEventsBackground / (1 - totalClassPercent))
    numAllTrainingEventsElevated = int(percentElevated * numAllTrainingEventsBackground / (1 - totalClassPercent))

    print("Num All Training events SEP: {}. Num All Training events Elevated: {}. Num All Training events Background {}.".format(numAllTrainingEventsSEP, numAllTrainingEventsElevated, numAllTrainingEventsBackground))
    
    # duplicate all training events
    selectionAllTrainingSEP = duplicate(allTrainingSEPSet, numAllTrainingEventsSEP)

    # duplicate all Elevated training events
    selectionAllTrainingElevated = duplicate(allTrainingElevatedSet, numAllTrainingEventsElevated)
    
    return (selectionTrainingSEP, validationSEPSet, selectionAllTrainingSEP, testSEPSet, selectionTrainingElevated, validationElevatedSet, selectionAllTrainingElevated, testElevatedSet, trainingBackgroundSet, validationBackgroundSet, allTrainingBackgroundSet, testBackgroundSet)

def create_first_stage_training(data):
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
        "Double_CME_100_MeV",
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

    training_percent = 0.55
    validation_percent = 0.15
    testPercent = 0.3
    secondLevelBool = "Double_CME_100_MeV"

    trainingSEP, validationSEP, testSEP = splitTrainingValidationTest(eventsSEP, training_percent, validation_percent, testPercent, secondLevelBool)
    trainingElevated, validationElevated, testElevated = splitTrainingValidationTest(eventsElevated, training_percent, validation_percent, testPercent, secondLevelBool)
    trainingBackground, validationBackground, testBackground = splitTrainingValidationTest(eventsBackground, training_percent, validation_percent, testPercent, secondLevelBool)

    writeToCSV(trainingSEP + trainingElevated + trainingBackground, "../res/gen/firstStageTraining.csv")
    writeToCSV(validationSEP + validationElevated + validationBackground, "../res/gen/firstStageValidation.csv")
    writeToCSV(trainingSEP + validationSEP + trainingElevated + validationElevated + trainingBackground + validationBackground, "../res/gen/firstStageAllTraining.csv")
    writeToCSV(testSEP + testElevated + testBackground, "../res/gen/firstStageTest.csv")

    writeToCSV(trainingSEP + trainingElevated, "../res/gen/syn/firstStageTrainingSyn.csv")
    writeToCSV(validationSEP + validationElevated, "../res/gen/syn/firstStageValidationSyn.csv")
    writeToCSV(trainingSEP + validationSEP + trainingElevated + validationElevated, "../res/gen/syn/firstStageAllTrainingSyn.csv")
    writeToCSV(testSEP + testElevated, "../res/gen/syn/firstStageTestSyn.csv")

    return (trainingSEP, validationSEP, testSEP, trainingElevated, validationElevated, testElevated, trainingBackground, validationBackground, testBackground)

def create_second_stage_training(data, trainingSEP, validationSEP, testSEP, trainingElevated, validationElevated, testElevated, trainingBackground, validationBackground, testBackground):
    for i in range(1, 10):
        percentSEP = (0.1 * i) / 2
        percentElevated = (0.1 * i) / 2
        trainingSEPSet, validationSEPSet, allTrainingSEPSet, testSEPSet, trainingElevatedSet, validationElevatedSet, allTrainingElevatedSet, testElevatedSet, trainingNonSEPSet, validationNonSEPSet, allTrainingNonSEPSet, testNonSEPSet = classbasedOversample(trainingSEP, validationSEP, testSEP, trainingElevated, validationElevated, testElevated, trainingBackground, validationBackground, testBackground, percentSEP = percentSEP, percentElevated = percentElevated)

        trainingFilename = "../res/gen/secondStageOversampleTraining_percentSEP_{:.1f}.csv".format((0.1 * i))
        print("WRITING TRAINING: {}".format(trainingFilename))
        writeToCSV(trainingSEPSet + trainingElevatedSet + trainingNonSEPSet, trainingFilename)

        validationFilename = "../res/gen/secondStageOversampleValidation_percentSEP_{:.1f}.csv".format((0.1 * i))
        print("WRITING VALIDATION: {}".format(validationFilename))
        writeToCSV(validationSEPSet + validationElevatedSet + validationNonSEPSet, validationFilename)

        allTrainingFilename = "../res/gen/secondStageOversampleAllTraining_percentSEP_{:.1f}.csv".format((0.1 * i))
        print("WRITING ALL TRAINING: {}".format(allTrainingFilename))
        writeToCSV(allTrainingSEPSet + allTrainingElevatedSet + allTrainingNonSEPSet, allTrainingFilename)

        testFilename = "../res/gen/secondStageOversampleTest_percentSEP_{:.1f}.csv".format((0.1 * i))
        print("WRITING TEST: {}".format(testFilename))
        writeToCSV(testSEPSet + testElevatedSet + testNonSEPSet, testFilename)

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

def getMatchingEvents(fromSet, toSet):
    mappedToSet = {}
    for event in toSet:
        mappedToSet[int(event["index"])] = event

    matchingEvents = []
    for event in fromSet:
        matchingEvents.append(mappedToSet[int(event["index"])])

    return matchingEvents

def create_stratified_richardson(sepEvents, elevatedEvents, trainingBackground, validationBackground, testBackground):
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

    for j in range(0, 3):
        writeToCSV(testSEPs[j] + testElevateds[j] + testBackground, folder + "/" + "firstStageTest_3Fold_{}.csv".format(j))
        writeToCSV(testSEPs[j] + testElevateds[j], syn_folder + "/" + "firstStageTest_3Fold_Syn_{}.csv".format(j))

def stratifyEvents(events, numTrainingPerBucket, numValidationPerBucket, numTestPerBucket):
    sortedEvents = sorted(events, key = lambda x: x["100MeV_peak_intensity"], reverse=True)
    
    trainingSet = list()
    validationSet = list()
    testSet = list()
        
    bucketSize = numTrainingPerBucket + numValidationPerBucket + numTestPerBucket
    for i in range(0, len(sortedEvents), bucketSize):
        nextBucketEvents = sortedEvents[i : i + bucketSize]
        
        randomSelection = random.sample(nextBucketEvents, len(nextBucketEvents))
            
        trainingSet.extend(randomSelection[0:numTrainingPerBucket])
        validationSet.extend(randomSelection[numTrainingPerBucket:numTrainingPerBucket + numValidationPerBucket])
        testSet.extend(randomSelection[numTrainingPerBucket + numValidationPerBucket:])

    return (trainingSet, validationSet, testSet)

def create_HARDCODED_stratified_first_stage_training(data):
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
        "Double_CME_100_MeV",
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

    # Training
    # 7 Elevated
    # 4 SEP
    # 1240 Background

    # Validation
    # 3 Elevated
    # 2 SEP
    # 340 Background

    # Test
    # 3 Elevated
    # 2 SEP
    # 676 Background
    
    numTrainingSEPEvents = 2
    numValidationSEPEvents = 1
    numTestSEPEvents = 1    
    trainingSEP, validationSEP, testSEP = stratifyEvents(eventsSEP, numTrainingSEPEvents, numValidationSEPEvents, numTestSEPEvents)

    numTrainingElevatedEvents = 7
    numValidationElevatedEvents = 3
    numTestElevatedEvents = 3

    sortedElevatedEvents = sorted(eventsElevated, key = lambda x: x["100MeV_peak_intensity"], reverse=True)

    # Bucket 1 Elevated
    numTrainingElevatedGroup1Events = 3
    numValidationElevatedGroup1Events = 1
    numTestElevatedGroup1Events = 1
    bucket1Events = sortedElevatedEvents[0:numTrainingElevatedGroup1Events + numValidationElevatedGroup1Events + numTestElevatedGroup1Events]
    trainingElevatedBucket1, validationElevatedBucket1, testElevatedBucket1 = stratifyEvents(bucket1Events, numTrainingElevatedGroup1Events, numValidationElevatedGroup1Events, numTestElevatedGroup1Events)

    # Bucket 2 Elevated
    numTrainingElevatedGroup2Events = 4
    numValidationElevatedGroup2Events = 2
    numTestElevatedGroup2Events = 2
    bucket2Events = sortedElevatedEvents[numTrainingElevatedGroup1Events + numValidationElevatedGroup1Events + numTestElevatedGroup1Events:]
    trainingElevatedBucket2, validationElevatedBucket2, testElevatedBucket2 = stratifyEvents(bucket2Events, numTrainingElevatedGroup2Events, numValidationElevatedGroup2Events, numTestElevatedGroup2Events)

    trainingElevated = trainingElevatedBucket1 + trainingElevatedBucket2
    validationElevated = validationElevatedBucket1 + validationElevatedBucket2
    testElevated = testElevatedBucket1 + testElevatedBucket2
    
    numTrainingBackgroundEvents = 1240
    numValidationBackgroundEvents = 340
    numTestBackgroundEvents = 676
    trainingBackground, validationBackground, testBackground = stratifyEvents(eventsBackground, numTrainingBackgroundEvents, numValidationBackgroundEvents, numTestBackgroundEvents)

    writeToCSV(trainingSEP + trainingElevated + trainingBackground, "../res/gen/firstStageTraining.csv")
    writeToCSV(validationSEP + validationElevated + validationBackground, "../res/gen/firstStageValidation.csv")
    writeToCSV(trainingSEP + validationSEP + trainingElevated + validationElevated + trainingBackground + validationBackground, "../res/gen/firstStageAllTraining.csv")
    writeToCSV(testSEP + testElevated + testBackground, "../res/gen/firstStageTest.csv")

    writeToCSV(trainingSEP + trainingElevated, "../res/gen/syn/firstStageTrainingSyn.csv")
    writeToCSV(validationSEP + validationElevated, "../res/gen/syn/firstStageValidationSyn.csv")
    writeToCSV(trainingSEP + validationSEP + trainingElevated + validationElevated, "../res/gen/syn/firstStageAllTrainingSyn.csv")
    writeToCSV(testSEP + testElevated, "../res/gen/syn/firstStageTestSyn.csv")

    return (trainingSEP, validationSEP, testSEP, trainingElevated, validationElevated, testElevated, trainingBackground, validationBackground, testBackground)


def main():
    data_file = "../res/adapted_rRT_data_learn_richardson.csv"
    data = readCSVFile(data_file)

    data = removeRows(data, feature="Double_CME_100_MeV", val=1)

    #trainingSEP, validationSEP, testSEP, trainingElevated, validationElevated, testElevated, trainingBackground, validationBackground, testBackground = create_first_stage_training(data)

    trainingSEP, validationSEP, testSEP, trainingElevated, validationElevated, testElevated, trainingBackground, validationBackground, testBackground = create_HARDCODED_stratified_first_stage_training(data)

    create_second_stage_training(data, trainingSEP, validationSEP, testSEP, trainingElevated, validationElevated, testElevated, trainingBackground, validationBackground, testBackground)

    normalized_seps = trainingSEP + validationSEP + testSEP
    normalized_elevated = trainingElevated + validationElevated + testElevated

    data_seps = getMatchingEvents(normalized_seps, data)
    data_elevated = getMatchingEvents(normalized_elevated, data)
    data_trainingBackground = getMatchingEvents(trainingBackground, data)
    data_validationBackground = getMatchingEvents(validationBackground, data)
    data_testBackground = getMatchingEvents(testBackground, data)
    create_stratified_richardson(data_seps, data_elevated, data_trainingBackground, data_validationBackground, data_testBackground)

if __name__ == "__main__":
    main()
