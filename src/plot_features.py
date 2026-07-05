from base_eval import *
import matplotlib.pyplot as plt
import math

def plotFeature(x, y, targets, of_interest, plotTitle, xLabel, yLabel, pngFilename, y_min, y_max):
    fig, ax = plt.subplots()
    ax.set_title(plotTitle)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    ax.set_ylim(y_min, y_max)

    fp_events_x = []
    fp_events_y = []
    scatterColors_fp_events = []
    marker_fp = "^"

    fn_events_x = []
    fn_events_y = []
    scatterColors_fn_events = []
    marker_fn = "v"

    scatters = []
    scatter_labels = ["Background", "Elevated", "SEP"]

    for particular_target in [0, 2, 1]:
        other_x = []
        other_y = []
        scatterColors_other_events = []
        marker_other = 'o'

        for i in range(0, len(targets)):
            target = targets[i]
            
            if target == particular_target:
                if of_interest[i] == 1:
                    fp_events_x.append(x[i])
                    fp_events_y.append(y[i])
                    scatterColors_fp_events.append("black")
                elif of_interest[i] == 2:
                    fp_events_x.append(x[i])
                    fp_events_y.append(y[i])
                    scatterColors_fp_events.append("black")
                elif of_interest[i] == 3:
                    fn_events_x.append(x[i])
                    fn_events_y.append(y[i])
                    scatterColors_fn_events.append("black")
                else:
                    color = "black"
                    if target == 0:
                        # Background
                        color = "blue"
                    elif target == 1:
                        # SEP
                        color = "red"
                    elif target == 2:
                        # Elevated
                        color = "green"

                    other_x.append(x[i])
                    other_y.append(y[i])
                    scatterColors_other_events.append(color)

        scatter = ax.scatter(other_x, other_y, c=scatterColors_other_events, marker=marker_other)
        scatters.append(scatter)

    if len(fp_events_x) > 0:
        scatter = ax.scatter(fp_events_x, fp_events_y, c=scatterColors_fp_events, marker=marker_fp)
        scatters.append(scatter)
        scatter_labels.append("FP")

    if len(fn_events_x) > 0:
        scatter = ax.scatter(fn_events_x, fn_events_y, c=scatterColors_fn_events, marker=marker_fn)
        scatters.append(scatter)
        scatter_labels.append("FN")

    ax.legend(scatters, scatter_labels)

    plt.savefig(pngFilename)
    plt.close(fig)

def symLogPlotFeature(x, y, targets, of_interest, plotTitle, xLabel, yLabel, pngFilename, y_min, y_max):
    fig, ax = plt.subplots()
    ax.set_title(plotTitle)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    ax.set_ylim(y_min, y_max)
    ax.set_xscale('symlog')

    fp_events_x = []
    fp_events_y = []
    scatterColors_fp_events = []
    marker_fp = "^"

    fn_events_x = []
    fn_events_y = []
    scatterColors_fn_events = []
    marker_fn = "v"

    scatters = []
    scatter_labels = ["Background", "Elevated", "SEP"]

    for particular_target in [0, 2, 1]:
        other_x = []
        other_y = []
        scatterColors_other_events = []
        marker_other = 'o'

        for i in range(0, len(targets)):
            target = targets[i]
            
            if target == particular_target:
                if of_interest[i] == 1:
                    fp_events_x.append(x[i])
                    fp_events_y.append(y[i])
                    scatterColors_fp_events.append("black")
                elif of_interest[i] == 2:
                    fp_events_x.append(x[i])
                    fp_events_y.append(y[i])
                    scatterColors_fp_events.append("black")
                elif of_interest[i] == 3:
                    fn_events_x.append(x[i])
                    fn_events_y.append(y[i])
                    scatterColors_fn_events.append("black")
                else:
                    color = "black"
                    if target == 0:
                        # Background
                        color = "blue"
                    elif target == 1:
                        # SEP
                        color = "red"
                    elif target == 2:
                        # Elevated
                        color = "green"

                    other_x.append(x[i])
                    other_y.append(y[i])
                    scatterColors_other_events.append(color)

        scatter = ax.scatter(other_x, other_y, c=scatterColors_other_events, marker=marker_other)
        scatters.append(scatter)

    if len(fp_events_x) > 0:
        scatter = ax.scatter(fp_events_x, fp_events_y, c=scatterColors_fp_events, marker=marker_fp)
        scatters.append(scatter)
        scatter_labels.append("FP")

    if len(fn_events_x) > 0:
        scatter = ax.scatter(fn_events_x, fn_events_y, c=scatterColors_fn_events, marker=marker_fn)
        scatters.append(scatter)
        scatter_labels.append("FN")

    ax.legend(scatters, scatter_labels)

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
        inputData_x.append(x_data)

        targets_x[i] = int(elem["target"])

        of_interest.append(int(elem["of_interest"]))
        
        y_data = dict()
        y_data["predicted_100MeV_peak_intensity_ln"] = float(elem["predicted_100MeV_peak_intensity_ln"])
        #y_data["100MeV_peak_intensity_ln"] = float(elem["100MeV_peak_intensity_ln"])
        data_y.append(y_data)
        
    return (inputData_x, targets_x, data_y, of_interest)

def getNonRichardsonFeatures(csvFile):
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
        x_data["V log V"] = float(elem["V log V"])
        x_data["halo"] = float(elem["halo"])
        x_data["diffusive_shock"] = float(elem["diffusive_shock"])
        inputData_x.append(x_data)

        targets_x[i] = int(elem["target"])

        of_interest.append(int(elem["of_interest"]))
        
        y_data = dict()
        y_data["predicted_100MeV_peak_intensity_ln"] = float(elem["predicted_100MeV_peak_intensity_ln"])
        #y_data["100MeV_peak_intensity_ln"] = float(elem["100MeV_peak_intensity_ln"])
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
        inputData_x.append(x_data)

        targets_x[i] = int(elem["target"])

        of_interest.append(int(elem["of_interest"]))
        
        y_data = dict()
        y_data["predicted_100MeV_peak_intensity_ln"] = float(elem["predicted_100MeV_peak_intensity_ln"])
        y_data["100MeV_peak_intensity_ln"] = float(elem["100MeV_peak_intensity_ln"])
        data_y.append(y_data)
        
    return (inputData_x, targets_x, data_y, of_interest)

def create_plots(data_x, targets_x, data_y, of_interest, yLabel, y_min, y_max, plot_title_prefix=''):
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

        if feature_name == "Accel":
            xLabel = "Acceleration Symlog Scale"
            plot_title_postfix = (yLabel + "_vs_" + xLabel + ".png").replace(" ", "_")
            symLogPlotFeature(collected_x_values, outputs["predicted_100MeV_peak_intensity_ln"], targets_x, of_interest, yLabel + " vs " + xLabel, xLabel, yLabel, "{}/{}".format(folder, plot_title_prefix + plot_title_postfix), y_min, y_max)
    
        if feature_name == "Type_2_Area":
            xLabel = "Type II Area Symlog Scale"
            plot_title_postfix = (yLabel + "_vs_" + xLabel + ".png").replace(" ", "_")
            symLogPlotFeature(collected_x_values, outputs["predicted_100MeV_peak_intensity_ln"], targets_x, of_interest, yLabel + " vs " + xLabel, xLabel, yLabel, "{}/{}".format(folder, plot_title_prefix + plot_title_postfix), y_min, y_max)

        for output_name in output_names:
            if feature_name == "donki_speed":
                xLabel = "Linear Speed"
            elif feature_name == "longitude":
                xLabel = "Longitude"
            elif feature_name == "latitude":
                xLabel = "Latitude"
            elif feature_name == "Accel":
                xLabel = "Acceleration"
            elif feature_name == "donki_ha":
                xLabel = "Half Width"
            elif feature_name == "CMEs_over_1000_past_9_hrs":
                xLabel = "CMEs over 1000 past 9 hrs"
            elif feature_name == "CMEs_past_month":
                xLabel = "Number of CMEs in the Past Month"
            elif feature_name == "2nd_order_speed_20R":
                xLabel = "2nd order speed at 20 solar radii"
            else:
                xLabel = feature_name
            plot_title_postfix = (yLabel + "_vs_" + xLabel + ".png").replace(" ", "_")
            plotFeature(collected_x_values, outputs[output_name], targets_x, of_interest, yLabel + " vs " + xLabel, xLabel, yLabel, "{}/{}".format(folder, plot_title_prefix + plot_title_postfix), y_min, y_max)

        if feature_name == "diffusive_shock":
            xLabel = "Diffusive Shock Log Scale"
            plot_title_postfix = (yLabel + "_vs_" + xLabel + ".png").replace(" ", "_")
            for i in range(0, len(collected_x_values)):
                collected_x_values[i] = math.log(collected_x_values[i])
            plotFeature(collected_x_values, outputs["predicted_100MeV_peak_intensity_ln"], targets_x, of_interest, yLabel + " vs " + xLabel, xLabel, yLabel, "{}/{}".format(folder, plot_title_prefix + plot_title_postfix), y_min, y_max)

        if feature_name == "Type_2_Area":
            xLabel = "Type II Area Log Scale"
            plot_title_postfix = (yLabel + "_vs_" + xLabel + ".png").replace(" ", "_")
            for i in range(0, len(collected_x_values)):
                if math.isclose(collected_x_values[i], 0):
                    collected_x_values[i] = collected_x_values[i]
                else:
                    collected_x_values[i] = math.log(collected_x_values[i])
            plotFeature(collected_x_values, outputs["predicted_100MeV_peak_intensity_ln"], targets_x, of_interest, yLabel + " vs " + xLabel, xLabel, yLabel, "{}/{}".format(folder, plot_title_prefix + plot_title_postfix), y_min, y_max)

#"..\\eval\\classifier\\retrained_autoencoder_ss_0_7\\threshold_0.6_Predictions_F1_0.8000000000000002_4.csv"
#"..\\eval\\classifier\\retrained_rRT_0_6\\threshold_0.6_Predictions_F1_0.6666666666666665.csv"
#"..\\eval\\classifier\\retrained_regNN_oversampled_0_2\\threshold_0.4_Predictions_F1_0.625.csv"
#min_y = -0.1
#max_y = 1.1

# "..\\eval\\denseLoss\\iterate\\alpha_0.90\\Predictions_F1_0.7272727272727272.csv"
# "..\\eval\\denseLoss_retrained_rRT\\alpha_0.70\\Predictions_F1_0.8000000000000002_4.csv"
# "..\\eval\\denseLoss_retrained_autoencoder_ss\\alpha_0.60\\Predictions_F1_0.8000000000000002_4.csv"

# "..\\eval\\using_training_richardson_alongside\\retrained_regNN_oversampled_0_6\\Predictions_F1_0.4.csv"
# "..\\eval\\using_training_richardson_alongside\\retrained_rRT_0_3\\Predictions_F1_0.5000000000000001_2.csv"
# "..\\eval\\using_training_richardson_alongside\\retrained_autoencoder_ss_0_2\\Predictions_F1_0.5454545454545454_4.csv"

# "..\\eval\\richardson_mixed\\retrained_regNN_oversampled_0_5\\Predictions_F1_0.22222222222222224.csv"
# "..\\eval\\richardson_mixed\\retrained_rRT_0_6\\Predictions_F1_0.22222222222222224_1.csv"
# "..\\eval\\richardson_mixed\\retrained_autoencoder_ss_0_5\\Predictions_F1_0.6666666666666666.csv"

# "..\\eval\\retrained_regNN_oversampled_0_7\\Predictions_F1_0.6_1.csv"
# "..\\eval\\retrained_rRT_0_1\\Predictions_F1_0.7272727272727272.csv"
# "..\\eval\\retrained_autoencoder_ss_0_1\\Predictions_F1_0.7272727272727272_3.csv

data_x, targets_x, data_y, of_interest = getFeatures("..\\eval\\denseLoss_retrained_autoencoder_ss\\alpha_0.60\\Predictions_F1_0.8000000000000002_4.csv")
#data_x, targets_x, data_y, of_interest = getNonRichardsonFeatures("..\\eval\\using_training_richardson_alongside\\retrained_autoencoder_ss_0_2\\Predictions_F1_0.5454545454545454_4.csv")

#data_x, targets_x, data_y, of_interest = getTrainedRichardsonFeatures("../eval/using_trained_richardson/retrained_regNN_oversampled_0_1/Predictions_F1_0.6666666666666666.csv")


min_y = min((-3.0, -2.4, -2.4, -3.2, -2.1, -2.7, -4.8, -4.0, -4.3, -2.3, -2.5, -2.4)) - 0.1
max_y = max((2.0, 0.7, 0.8, 1.8, 2.4, 1.1, 4.4, 1.7, 1.9, 1.1, 0.8, 0.6)) + 0.1

#create_plots(data_x, targets_x, data_y, of_interest, "Score", min_y, max_y)
create_plots(data_x, targets_x, data_y, of_interest, "Predicted Peak Intensity LN", min_y, max_y)




def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def denseWeightPlotting():
    x = []
    y = []
    targets = []
    features_with_dense_loss = readCSVFile("../denseLoss/features_with_dense_loss.csv")
    for elem in features_with_dense_loss:
        intensity = float(elem["100MeV_peak_intensity_ln"])
        x.append(intensity)

        y.append(float(elem["dense_loss_weighting_func_alpha_0.60"]))

        targets.append(float(elem["target"]))

    min_x = min(x) - 0.1
    max_x = max(x) + 0.1
    min_y = min(y) - 0.1
    max_y = max(y) + 0.1

    fig, ax = plt.subplots()
    ax.set_title("Dense Weight for DW = 0.6 vs Actual Peak Intensity LN")
    ax.set_xlabel("Actual Peak Intensity LN")
    ax.set_ylabel("Dense Weight for DW = 0.6")

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    scatters = []
    scatter_labels = ["Background", "Elevated", "SEP"]

    for particular_target in [0, 2, 1]:
        other_x = []
        other_y = []
        scatterColors_other_events = []
        marker_other = 'o'

        for i in range(0, len(targets)):
            target = targets[i]
            
            if target == particular_target:
                color = "black"
                if target == 0:
                    # Background
                    color = "blue"
                elif target == 1:
                    # SEP
                    color = "red"
                elif target == 2:
                    # Elevated
                    color = "green"

                other_x.append(x[i])
                other_y.append(y[i])
                scatterColors_other_events.append(color)

        scatter = ax.scatter(other_x, other_y, c=scatterColors_other_events, marker=marker_other)
        scatters.append(scatter)

    ax.legend(scatters, scatter_labels)

    plt.savefig("..\\eval\\Dense_Weight_0_6_vs_Actual_Peak_Intensity_LN.png")
    plt.close(fig)
