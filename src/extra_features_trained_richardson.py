from email.utils import parsedate
import numpy as np
import csv
import math
import sys
from astropy.time import Time

def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def writeCSVFile(data, csvFilename, fieldnames):
    with open(csvFilename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()        
        
        for elem in data:
            row = {}
            for feature in fieldnames:
                row[feature] = elem[feature]
            writer.writerow(row)
    return


def applyFuncToFeature(data, feature, func):
    for elem in data:
        elem[feature] = func(elem[feature])

def convertToFloat(val):
    try:
        return float(val)
    except:
        print("FAILED TO CONVERT TO FLOAT: {}".format(val), file=sys.stderr)
        return val

def calculate_learned_richardson(data, connection_angle_feature_name, newFeatureName):
    # Original richardson
    # I = 0.013 * EXP(0.0036 * V - connection_angle^2 / (2 * sigma^2))
    # LN(I) = LN(0.013 * EXP(0.0036 * V - connection_angle^2 / (2 * sigma^2)))
    # LN(I) = LN(0.013) + LN(EXP(0.0036 * V - connection_angle^2 / (2 * sigma^2)))
    # LN(I) = LN(0.013) + 0.0036 * V - connection_angle^2 / (2 * sigma^2)
    # LN(I) = LN(w_exp) + w_v * V - connection_angle^2 / (2 * sigma^2)
    # Now, we have new values for w_esp and w_v

    learned_richardson_coefficents = readCSVFile("../res/trained/richardson.csv")
    w_exp = float(learned_richardson_coefficents[0]["w_exp"])
    w_v = float(learned_richardson_coefficents[0]["w_v"])

    for elem in data:
        V = float(elem["donki_speed"])
        connection_angle = float(elem[connection_angle_feature_name])
        richardson_intensity_ln = math.log(w_exp) + w_v * V - math.pow(connection_angle, 2) / (2 * math.pow(43, 2))
        elem[newFeatureName] = richardson_intensity_ln

def main():
    data = readCSVFile("../res/adapted_rRT_data_learn_richardson.csv")
    
    fieldnames = []
    for key in data[0].keys():
        fieldnames.append(key)

    trained_richardson_ln_feature_name = "trained_richardson_ln"
    calculate_learned_richardson(data, "connection_angle_degrees", trained_richardson_ln_feature_name)
    fieldnames.append(trained_richardson_ln_feature_name)

    writeCSVFile(data, "../res/adapted_rRT_data_learn_richardson.csv", fieldnames)

if __name__ == "__main__":
    main()
