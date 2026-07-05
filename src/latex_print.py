import copy
import csv

def results():
    s = "\
    cRegNN 20 & 0.4 & 5.8 & 0.4 & 4.6 & 675.2 & \\underline{0.598} & \\underline{0.594} & 0.911\\\\\
    cRT 60 & 0.6 & \\underline{3.2} & 1.2 & 3.8 & \\underline{677.8} & \\underline{0.641} & \\underline{0.638} & 0.755\\\\\
    cRT+AE 70 & 0.6 & \\underline{1.0} & 1.0 & 4.0 & \\underline{680.0} & \\textbf{\\underline{0.800}} & \\textbf{\\underline{0.799}} & 0.799\\\\\
    "


    pieces = s.split("\\\\")

    while len(pieces[-1]) == 0:
        pieces.pop()

    sub_pieces = []
    for row in pieces:
        splitted_row = row.split("&")
        sub_pieces_row = []
        for str_val in splitted_row:
            stripped_str_val = str_val.strip()
            if stripped_str_val == "":
                continue

            start_textbf = "\\textbf{"
            if stripped_str_val.startswith(start_textbf):
                stripped_str_val = stripped_str_val[len(start_textbf):]
            start_underline = "\\underline{"
            if stripped_str_val.startswith(start_underline):
                stripped_str_val = stripped_str_val[len(start_underline):]

            while len(stripped_str_val) > 0 and stripped_str_val[-1] == "}":
                stripped_str_val = stripped_str_val[:-1]

            sub_pieces_row.append(stripped_str_val.strip())
        if len(sub_pieces_row) > 0:
            sub_pieces.append(sub_pieces_row)

    min_indexes = [0] * len(sub_pieces[0])
    max_indexes = [0] * len(sub_pieces[0])

    #extract = [None, "max", "max", "min", "min"]
    #extract = [None, "min", "min", "max", "max", "max", "max", "max"]
    extract = [None, None, "min", "min", "max", "max", "max", "max", "max"]

    min_row_index = {}
    max_row_index = {}
    for index in range(0, len(sub_pieces)):
        row_name = sub_pieces[index][0].split(' ')[0]

        if row_name not in min_row_index:
            min_row_index[row_name] = [index] * len(sub_pieces[0])
        if row_name not in max_row_index:
            max_row_index[row_name] = [index] * len(sub_pieces[0])

        for j in range(0, len(sub_pieces[0])):
            if extract[j] is None:
                continue

            if float(sub_pieces[index][j]) < float(sub_pieces[min_row_index[row_name][j]][j]):
                min_row_index[row_name][j] = index
            if float(sub_pieces[index][j]) > float(sub_pieces[max_row_index[row_name][j]][j]):
                max_row_index[row_name][j] = index

            if float(sub_pieces[index][j]) < float(sub_pieces[min_indexes[j]][j]):
                min_indexes[j] = index
            if float(sub_pieces[index][j]) > float(sub_pieces[max_indexes[j]][j]):
                max_indexes[j] = index

    best_row_index = {}
    for key in min_row_index.keys():
        best_row_index[key] = [0] * len(sub_pieces[0])

    best_indexes = []

    for i in range(0, len(extract)):
        elem = extract[i]
        if elem is None:
            best_indexes.append(None)
            for key in best_row_index.keys():
                best_row_index[key][i] = None
        elif elem == "max":
            best_indexes.append(max_indexes[i])
            for key in best_row_index.keys():
                best_row_index[key][i] = max_row_index[key][i]
        elif elem == "min":
            best_indexes.append(min_indexes[i])
            for key in best_row_index.keys():
                best_row_index[key][i] = min_row_index[key][i]

    mod_sub_pieces = copy.deepcopy(sub_pieces)

    for index in range(0, len(sub_pieces)):
        row_name = sub_pieces[index][0].split(' ')[0]

        best_row_index_list = best_row_index[row_name]

        for j in range(0, len(best_row_index_list)):
            if extract[j] is None:
                continue
            if abs(float(sub_pieces[index][j]) - float(sub_pieces[best_row_index_list[j]][j])) < 1e-5:
                mod_sub_pieces[index][j] = "\\underline{" + mod_sub_pieces[index][j] + "}"


        for j in range(0, len(best_indexes)):
            if extract[j] is None:
                continue
            if abs(float(sub_pieces[index][j]) - float(sub_pieces[best_indexes[j]][j])) < 1e-5:
                mod_sub_pieces[index][j] = "\\textbf{" + mod_sub_pieces[index][j] + "}"

    last_row_name = None
    formatted_str = ""
    for index in range(0, len(mod_sub_pieces)):
        row_name = mod_sub_pieces[index][0].split(' ')[0]
        if last_row_name is None:
            last_row_name = row_name
        elif row_name != last_row_name:
            last_row_name = row_name
            formatted_str = formatted_str + "\\midrule\n"

        formatted_str = formatted_str + " & ".join(mod_sub_pieces[index]) + "\\\\" + "\n"

    print(formatted_str)

def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def analysis_features():
    filename = "..\\eval\\classifier\\retrained_rRT_0_6\\threshold_0.6_Predictions_F1_0.6666666666666665.csv"
    data = readCSVFile(filename)
    fp_elems = []
    fn_elems = []
    for elem in data:
        interest = int(elem["of_interest"])
        if interest == 1:
            fp_elems.append(elem)
        if interest == 3:
            fn_elems.append(elem)

    features = ["donki_date", "cdaw_date", "latitude", "longitude", "donki_speed", "Accel", "100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln", "predicted_thresholded"]
    row_names = ["DONKI Date", "CDAW Date", "Latitude", "Longitude", "Linear Speed", "Acceleration", "100 MeV Peak Intensity LN", "Classifier Score", "Classifier Prediction"]
    rows = [""] * len(features)
    header = "Feature"

    for i in range(0, len(rows)):
        rows[i] = rows[i] + row_names[i]
    
    for elem in fp_elems:
        header = header + " & FP"
        for i in range(0, len(rows)):
            rows[i] = rows[i] + " & "
            if features[i] == "predicted_thresholded":
                if int(elem[features[i]]) == 1:
                    rows[i] = rows[i] + "SEP"
                else:
                    rows[i] = rows[i] + "Non-SEP"
            elif features[i] == "Accel":
                rows[i] = rows[i] + str(round(float(elem[features[i]])))
            elif features[i] in ["100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln"]:
                rows[i] = rows[i] + str(round(float(elem[features[i]]), 3))
            else:
                rows[i] = rows[i] + elem[features[i]]

    for elem in fn_elems:
        header = header + " & FN"
        for i in range(0, len(rows)):
            rows[i] = rows[i] + " & "
            if features[i] == "predicted_thresholded":
                if int(elem[features[i]]) == 1:
                    rows[i] = rows[i] + "SEP"
                else:
                    rows[i] = rows[i] + "Non-SEP"
            elif features[i] == "Accel":
                rows[i] = rows[i] + str(round(float(elem[features[i]])))
            elif features[i] in ["100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln"]:
                rows[i] = rows[i] + str(round(float(elem[features[i]]), 3))
            else:
                rows[i] = rows[i] + elem[features[i]]

    header = header + "\\\\"
    for i in range(0, len(rows)):
        rows[i] = rows[i] + "\\\\"

    s = header + "\n" + "\\midrule" + "\n"
    for row in rows:
        s = s + row + "\n"

    print(s)

def analysis_features_alt():
    #"..\\eval\\classifier\\retrained_regNN_oversampled_0_2\\threshold_0.4_Predictions_F1_0.625.csv"
    #"..\\eval\\classifier\\retrained_rRT_0_6\\threshold_0.6_Predictions_F1_0.6666666666666665.csv"
    #"..\\eval\\classifier\\retrained_autoencoder_ss_0_7\\threshold_0.6_Predictions_F1_0.8000000000000002_4.csv"

    filename = "..\\eval\\classifier\\retrained_rRT_0_6\\threshold_0.6_Predictions_F1_0.6666666666666665.csv"
    data = readCSVFile(filename)
    fp_elems = []
    fn_elems = []
    for elem in data:
        interest = int(elem["of_interest"])
        if interest == 1 or interest == 2:
            fp_elems.append(elem)
        if interest == 3:
            fn_elems.append(elem)

    header = ""
    col_names = ["DD", "Lat", "Lon", "Spd", "Ha", "Acc", "Peak LN", "Score", "FP/FN"]
    for col in col_names:
        if len(header) > 0:
            header = header + " & "
        header = header + col
    header = header + "\\\\"

    features = ["donki_date", "latitude", "longitude", "donki_speed", "donki_ha", "Accel", "100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln", "predicted_thresholded"]
    rows = [""] * (len(fp_elems) + len(fn_elems))

    row_index = 0
    
    for elem in fp_elems:
        for feature in features:
            if len(rows[row_index]) > 0:
                rows[row_index] = rows[row_index] + " & "
            if feature == "predicted_thresholded":
                if int(elem[feature]) == 1:
                    rows[row_index] = rows[row_index] + "FP"
                else:
                    rows[row_index] = rows[row_index] + "FN"
            elif feature == "Accel":
                rows[row_index] = rows[row_index] + "{:.1f}".format(float(elem[feature]))
            elif feature in ["100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln"]:
                rows[row_index] = rows[row_index] + "{:.3f}".format(float(elem[feature]))
            else:
                rows[row_index] = rows[row_index] + elem[feature]
        row_index = row_index + 1
           
    for elem in fn_elems:
        for feature in features:
            if len(rows[row_index]) > 0:
                rows[row_index] = rows[row_index] + " & "
            if feature == "predicted_thresholded":
                if int(elem[feature]) == 1:
                    rows[row_index] = rows[row_index] + "FP"
                else:
                    rows[row_index] = rows[row_index] + "FN"
            elif feature == "Accel":
                rows[row_index] = rows[row_index] + "{:.1f}".format(float(elem[feature]))
            elif feature in ["100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln"]:
                rows[row_index] = rows[row_index] + "{:.3f}".format(float(elem[feature]))
            else:
                rows[row_index] = rows[row_index] + elem[feature]
        row_index = row_index + 1

    for i in range(0, len(rows)):
        rows[i] = rows[i] + "\\\\"

    s = header + "\n" + "\\midrule" + "\n"
    for row in rows:
        s = s + row + "\n"

    print(s)

def regression_analysis_features_alt():
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

    filename = "..\\eval\\denseLoss_retrained_autoencoder_ss\\alpha_0.60\\Predictions_F1_0.8000000000000002_4.csv"
    data = readCSVFile(filename)
    fp_elems = []
    fn_elems = []
    for elem in data:
        interest = int(elem["of_interest"])
        if interest == 1 or interest == 2:
            fp_elems.append(elem)
        if interest == 3:
            fn_elems.append(elem)

    header = ""
    col_names = ["DD", "Lat", "Lon", "Spd", "Ha", "Acc", "Peak LN", "Pred", "FP/FN"]
    for col in col_names:
        if len(header) > 0:
            header = header + " & "
        header = header + col
    header = header + "\\\\"

    features = ["donki_date", "latitude", "longitude", "donki_speed", "donki_ha", "Accel", "100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln"]
    rows = [""] * (len(fp_elems) + len(fn_elems))

    row_index = 0
    
    for elem in fp_elems:
        for feature in features:
            if len(rows[row_index]) > 0:
                rows[row_index] = rows[row_index] + " & "
            if feature in ["100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln"]:
                rows[row_index] = rows[row_index] + "{:.3f}".format(float(elem[feature]))
                if feature == "predicted_100MeV_peak_intensity_ln":
                    if float(elem[feature]) >= 0:
                        rows[row_index] = rows[row_index] + " & FP"
                    else:
                        rows[row_index] = rows[row_index] + " & FN"
            elif feature == "Accel":
                rows[row_index] = rows[row_index] + "{:.1f}".format(float(elem[feature]))
            else:
                rows[row_index] = rows[row_index] + elem[feature]
        row_index = row_index + 1
           
    for elem in fn_elems:
        for feature in features:
            if len(rows[row_index]) > 0:
                rows[row_index] = rows[row_index] + " & "
            if feature in ["100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln"]:
                rows[row_index] = rows[row_index] + "{:.3f}".format(float(elem[feature]))
                if feature == "predicted_100MeV_peak_intensity_ln":
                    if float(elem[feature]) >= 0:
                        rows[row_index] = rows[row_index] + " & FP"
                    else:
                        rows[row_index] = rows[row_index] + " & FN"
            elif feature == "Accel":
                rows[row_index] = rows[row_index] + "{:.1f}".format(float(elem[feature]))
            else:
                rows[row_index] = rows[row_index] + elem[feature]
        row_index = row_index + 1

    for i in range(0, len(rows)):
        rows[i] = rows[i] + "\\\\"

    s = header + "\n" + "\\midrule" + "\n"
    for row in rows:
        s = s + row + "\n"

    print(s)

analysis_features_alt()