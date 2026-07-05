from base_func import *

def plotFeature(x, y, targets, of_interest, plotTitle, xLabel, yLabel, pngFilename):
    scatterColors = []
    for i in range(0, len(targets)):
        target = targets[i]
        color = "black"
        if of_interest[i] == 1:
            color = "orange"
        elif of_interest[i] == 2:
            color = "violet"
        elif of_interest[i] == 3:
            color = "cyan"
        elif of_interest[i] == 4:
            color = "brown"
        else:
            if target == 0:
                # Background
                color = "blue"
            elif target == 1:
                # SEP
                color = "red"
            elif target == 2:
                # Elevated
                color = "green"
        scatterColors.append(color)

    fig, ax = plt.subplots()

    scatter = ax.scatter(x, y, c=scatterColors)
    ax.set_title(plotTitle)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    plt.savefig(pngFilename)
    plt.close(fig)

def readFeatures(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def getFeatures(csvFile):
    data = readCSVFile(csvFile)
    
    numElements = len(data)
    inputData_x = list()
    targets_x = np.empty(numElements, dtype=int)
    of_interest = list()
    data_y = list()
    
    for i in range(0, numElements):
        elem = data[i]
        
        x_data = dict()
        x_data["donki_speed"] = float(elem["donki_speed"])
        x_data["donki_ha"] = float(elem["donki_ha"])
        x_data["longitude"] = float(elem["longitude"])
        x_data["latitude"] = float(elem["latitude"])
        x_data["Accel"] = float(elem["Accel"])
        x_data["Type_2_Area"] = float(elem["Type_2_Area"])
        x_data["2nd_order_speed_final"] = float(elem["2nd_order_speed_final"])
        x_data["2nd_order_speed_20R"] = float(elem["2nd_order_speed_20R"])
        x_data["Central_PA"] = float(elem["Central_PA"])
        x_data["MPA"] = float(elem["MPA"])
        x_data["sunspots"] = float(elem["sunspots"])
        x_data["CMEs_past_month"] = float(elem["CMEs_past_month"])
        x_data["CMEs_past_9_hours"] = float(elem["CMEs_past_9_hours"])
        x_data["CMEs_over_1000_past_9_hrs"] = float(elem["CMEs_over_1000_past_9_hrs"])
        x_data["Max_speed_past_day"] = float(elem["Max_speed_past_day"])
        x_data["richardson_formula_degrees_phi_2_solar_wind"] = float(elem["richardson_formula_degrees_phi_2_solar_wind"])
        x_data["V log V"] = float(elem["V log V"])
        x_data["halo"] = float(elem["halo"])
        x_data["diffusive_shock"] = float(elem["diffusive_shock"])
        x_data["Double_CME_100_MeV"] = float(elem["Double_CME_100_MeV"])
        inputData_x.append(x_data)

        targets_x[i] = int(elem["target"])

        of_interest.append(int(elem["of_interest"]))
        
        y_data = dict()
        y_data["predicted_100MeV_peak_intensity_ln"] = float(elem["predicted_100MeV_peak_intensity_ln"])
        y_data["100MeV_peak_intensity_ln"] = float(elem["100MeV_peak_intensity_ln"])
        data_y.append(y_data)
        
    return (inputData_x, targets_x, data_y, of_interest)

def getTrainedRichardsonFeatures(csvFile):
    data = readCSVFile(csvFile)
    
    numElements = len(data)
    inputData_x = list()
    targets_x = np.empty(numElements, dtype=int)
    of_interest = list()
    data_y = list()
    
    for i in range(0, numElements):
        elem = data[i]
        
        x_data = dict()
        x_data["donki_speed"] = float(elem["donki_speed"])
        x_data["donki_ha"] = float(elem["donki_ha"])
        x_data["longitude"] = float(elem["longitude"])
        x_data["latitude"] = float(elem["latitude"])
        x_data["Accel"] = float(elem["Accel"])
        x_data["Type_2_Area"] = float(elem["Type_2_Area"])
        x_data["2nd_order_speed_final"] = float(elem["2nd_order_speed_final"])
        x_data["2nd_order_speed_20R"] = float(elem["2nd_order_speed_20R"])
        x_data["Central_PA"] = float(elem["Central_PA"])
        x_data["MPA"] = float(elem["MPA"])
        x_data["sunspots"] = float(elem["sunspots"])
        x_data["CMEs_past_month"] = float(elem["CMEs_past_month"])
        x_data["CMEs_past_9_hours"] = float(elem["CMEs_past_9_hours"])
        x_data["CMEs_over_1000_past_9_hrs"] = float(elem["CMEs_over_1000_past_9_hrs"])
        x_data["Max_speed_past_day"] = float(elem["Max_speed_past_day"])
        x_data["trained_richardson_ln"] = float(elem["trained_richardson_ln"])
        x_data["V log V"] = float(elem["V log V"])
        x_data["halo"] = float(elem["halo"])
        x_data["diffusive_shock"] = float(elem["diffusive_shock"])
        x_data["Double_CME_100_MeV"] = float(elem["Double_CME_100_MeV"])
        inputData_x.append(x_data)

        targets_x[i] = int(elem["target"])

        of_interest.append(int(elem["of_interest"]))
        
        y_data = dict()
        y_data["predicted_100MeV_peak_intensity_ln"] = float(elem["predicted_100MeV_peak_intensity_ln"])
        y_data["100MeV_peak_intensity_ln"] = float(elem["100MeV_peak_intensity_ln"])
        data_y.append(y_data)
        
    return (inputData_x, targets_x, data_y, of_interest)

def create_plots(data_x, targets_x, data_y, of_interest, plot_title_prefix=''):
    output_names = data_y[0].keys()
    outputs = dict()
    for output_name in output_names:
        collected_y_values = np.empty(len(data_y), dtype=np.float64)
        for i in range(0, len(data_y)):
            collected_y_values[i] = data_y[i][output_name]
        outputs[output_name] = collected_y_values

    folder = "../eval/features"
    feature_names = data_x[0].keys()
    for feature_name in feature_names:
        collected_x_values = np.empty(len(data_x), dtype=np.float64)
        for i in range(0, len(data_x)):
            collected_x_values[i] = data_x[i][feature_name]
    
        for output_name in output_names:
            xLabel = feature_name
            yLabel = output_name
            plotFeature(collected_x_values, outputs[output_name], targets_x, of_interest, yLabel + " vs " + xLabel, xLabel, yLabel, "{}/{}".format(folder, plot_title_prefix + yLabel + "_vs_" + xLabel + ".png"))

#data_x, targets_x, data_y, of_interest = getFeatures("../eval/retrained_autoencoder_ss_0_3/Predictions_F1_0.6666666666666666.csv")

data_x, targets_x, data_y, of_interest = getTrainedRichardsonFeatures("../eval/using_trained_richardson/retrained_autoencoder_ss_0_6/Predictions_F1_0.5_2.csv")


create_plots(data_x, targets_x, data_y, of_interest)

