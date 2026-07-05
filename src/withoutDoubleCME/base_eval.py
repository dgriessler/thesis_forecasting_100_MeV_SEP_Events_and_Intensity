import csv
import numpy as np

def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def get_indexes(csv_filename):
    data = list()
    data_dict = readCSVFile(csv_filename)
    for elem in data_dict:
        try:
            index = int(elem["index"])
            data.append(index)
        except:
            print("ERROR: INVALID DATA: {}. EXITING EARLY".format(elem))
            break
    return data

def get_real_data(origData, indexes):
    real_data = list()
    for index in indexes:
        idx = int(index)
        real_data.append(origData[idx])
    return real_data

def get_orig_data():
    data_dict = readCSVFile("../res/adapted_rRT_data_learn_richardson.csv")
    orig_data = dict()
    for elem in data_dict:
        index = int(elem["index"])
        orig_data[index] = elem
    return orig_data

def getFilename(header, F1, seenF1Scores, trailer=""):
    actualF1 = F1
    if F1 in seenF1Scores:
        actualF1 = str(F1) + "_" + str(seenF1Scores[F1])
        seenF1Scores[F1] = seenF1Scores[F1] + 1
    else:
        seenF1Scores[F1] = 1
    return "{}{}{}".format(header, actualF1, trailer)

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))