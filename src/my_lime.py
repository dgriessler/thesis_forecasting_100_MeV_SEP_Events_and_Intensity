from __future__ import print_function
from ast import Lambda
from audioop import reverse
from base64 import encode
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import csv
from regression import *
from classifier import *

def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def dl_rrt_ae_0_6():
        outFolder = "../out/denseLoss_retrained_autoencoder_ss"       

        dense_alpha = 0.6
        adamLearningRate = 0.001
        adam_epsilon = 1.0
        alpha = 0.3

        weights_filename = "{}/alpha_{:.2f}_it_{}.ckpt".format(outFolder, dense_alpha, 4)
        print("WEIGHTS_FILENAME: {}".format(weights_filename))

        autoencoder_validation_weights_path = "../out/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
        adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)
        adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)

        reg = Regression()
        autoencoder_second_stage_model = reg.denseLoss_autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, None, None, None, None, dense_alpha, alpha, weights_filename=weights_filename, train=False, seed=1234)
        
        return autoencoder_second_stage_model

dl_rrt_ae_0_6_projection = dl_rrt_ae_0_6()

def dl_rrt_ae_0_6_predict_proba(X):
    projection = dl_rrt_ae_0_6_projection
    out = projection.predict(X)
    arr = np.array(out)
    return arr

def crt_ae_70():
    folder_trailer = int(70 / 10)
    outFolder = "../out/classifier/retrained_autoencoder_ss_0_{}".format(folder_trailer)

    adamLearningRate = 0.001
    adamEpsilon = 1.0
    alpha = 0.3

    weights_filename = "{}/it_{}.ckpt".format(outFolder, 4)
    print("WEIGHTS_FILENAME: {}".format(weights_filename))

    autoencoder_validation_weights_path = "../out/classifier/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
    adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)
    adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adamEpsilon)

    cla = Classifier()
    autoencoder_second_stage_model = cla.autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, None, None, None, None, alpha, weights_filename=weights_filename, train=False, seed=1234)
    return autoencoder_second_stage_model 

crt_ae_70_projection = crt_ae_70()

def crt_ae_70_predict_proba(X):
    projection = crt_ae_70_projection
    out = projection.predict(X)
    arr = []
    for row in out:
        arr.append([1-row[0], row[0]])
    arr = np.array(arr)
    return arr

def collect_data(): 
    train_data = readCSVFile("../res/gen/firstStageAllTraining.csv")
    test_data = readCSVFile("../res/gen/firstStageTest.csv")

    index_map = {}
    reverse_index_map = {}

    all_data = []
    index = 0
    for elem in train_data + test_data:
        new_row = []
        for feature_name in feature_names:
            try:
                new_row.append(float(elem[feature_name]))
            except:
                print("FAILED: {} {}".format(feature_name, elem[feature_name]))
        all_data.append(new_row)
        index_map[int(elem["index"])] = index
        reverse_index_map[index] = int(elem["index"])
        index = index + 1

    all_data = np.array(all_data)

    training_data = []
    for elem in train_data:
        new_row = []
        for feature_name in feature_names:
            try:
                new_row.append(float(elem[feature_name]))
            except:
                print("FAILED: {} {}".format(feature_name, elem[feature_name]))
        training_data.append(new_row)

    training_data = np.array(training_data)

    return (index_map, all_data, reverse_index_map, training_data)

def get_explanation_list_dl_rrt_ae_0_6(explainer, event, num_features):
    exp = explainer.explain_instance(event, dl_rrt_ae_0_6_predict_proba, num_features=num_features)
    exp_list = exp.as_list()
    return exp_list

def get_explanation_list_crt_ae_70(explainer, event, num_features):
    exp = explainer.explain_instance(event, crt_ae_70_predict_proba, num_features=num_features)
    exp_list = exp.as_list()
    return exp_list

def normalize_feature_importance(exp_list):
    min_feature_importance = min(exp_list, key=lambda x : x[1])
    max_feature_importance = max(exp_list, key=lambda x : x[1])
    normalize_exp_list = []

    normalize_exp_list = [(x[0], (x[1] - min_feature_importance[1]) / (max_feature_importance[1] - min_feature_importance[1])) for x in exp_list]
    normalize_exp_list.sort(reverse=True, key=lambda x : x[1])
    return normalize_exp_list

def get_feature_metrics(explainer, event, feature_names, get_explanation_list_func):
    metrics = {}
    for feature_name in feature_names:
        metrics[feature_name] = 0.0

    exp_list = get_explanation_list_func(explainer, event, len(feature_names))
    for elem in exp_list:
        elem_feature_name_val = elem[0]
        found_feature_name = None
        for feature_name in feature_names:
            if feature_name in elem_feature_name_val:
                found_feature_name = feature_name
                break
        if found_feature_name is None:
            print("ERROR: CANNOT FIND FEATURE NAME: {}".format(elem_feature_name_val))
        metrics[found_feature_name] = metrics[found_feature_name] + elem[1]

    return metrics

def get_feature_metrics_full_details(explainer, event, feature_names, get_explanation_list_func):
    metrics = {}
    for feature_name in feature_names:
        metrics[feature_name] = ""

    exp_list = get_explanation_list_func(explainer, event, len(feature_names))
    for elem in exp_list:
        elem_feature_name_val = elem[0]
        found_feature_name = None
        for feature_name in feature_names:
            if feature_name in elem_feature_name_val:
                found_feature_name = feature_name
                break
        if found_feature_name is None:
            print("ERROR: CANNOT FIND FEATURE NAME: {}".format(elem_feature_name_val))
        metrics[found_feature_name] = metrics[found_feature_name] + str(elem)

    return metrics

def get_all_metrics_for_seed(explainer, seed, all_data, reverse_index_map, feature_names, get_explanation_list_func):
    rows = []
    for i in range(0, len(all_data)):
        event = all_data[i]
        w_i = get_feature_metrics(explainer, event, feature_names, get_explanation_list_func)
        row = {
            "Seed": seed,
            "Event Index": reverse_index_map[i]
        }
        for feature_name in feature_names:
            row[feature_name] = w_i[feature_name]
        rows.append(row)

    return rows

def get_all_metrics_full_details_for_seed(explainer, seed, all_data, reverse_index_map, feature_names, get_explanation_list_func):
    rows = []
    for i in range(0, len(all_data)):
        event = all_data[i]
        w_i = get_feature_metrics_full_details(explainer, event, feature_names, get_explanation_list_func)
        row = {
            "Seed": seed,
            "Event Index": reverse_index_map[i]
        }
        for feature_name in feature_names:
            row[feature_name] = w_i[feature_name]
        rows.append(row)

    return rows

def get_all_metrics_crt(training_data, all_data, reverse_index_map, feature_names, class_names, get_all_metrics_seed_func):
    header_names = []
    header_names.append("Seed")
    header_names.append("Event Index")
    for feature_name in feature_names:
        header_names.append(feature_name)
    
    seeds = [1234, 1235, 1236, 1237, 1238]
    all_rows = []
    for seed in seeds:
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data, feature_names=feature_names, categorical_features=[17], class_names=class_names, discretize_continuous=False, random_state=seed)
        seed_rows = get_all_metrics_seed_func(explainer, seed, all_data, reverse_index_map, feature_names, get_explanation_list_crt_ae_70)
        all_rows.extend(seed_rows)

    return (header_names, all_rows)

def get_all_metrics_dl(training_data, all_data, reverse_index_map, feature_names, get_all_metrics_seed_func):
    header_names = []
    header_names.append("Seed")
    header_names.append("Event Index")
    for feature_name in feature_names:
        header_names.append(feature_name)
    
    seeds = [1234, 1235, 1236, 1237, 1238]
    all_rows = []
    for seed in seeds:
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data, feature_names=feature_names, categorical_features=[17], class_names=["SEP"], discretize_continuous=False, random_state=seed, mode="regression")
        seed_rows = get_all_metrics_seed_func(explainer, seed, all_data, reverse_index_map, feature_names, get_explanation_list_dl_rrt_ae_0_6)
        all_rows.extend(seed_rows)

    return (header_names, all_rows)

def output_metrics(csvFilename, header_names, all_rows):
    with open(csvFilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header_names)
            writer.writeheader()

            for row in all_rows:
                writer.writerow(row)
    return

def get_indexes(csvFilename):
    data = readCSVFile(csvFilename)
    seed = int(data[0]["Seed"])
    indexes = []
    for elem in data:
        if int(elem["Seed"]) == seed:
            indexes.append(int(elem["Event Index"]))
    return indexes

def get_num_seeds(csvFilename):
    data = readCSVFile(csvFilename)   
    seeds_found = []
    for elem in data:
        if int(elem["Seed"]) not in seeds_found:
            seeds_found.append(int(elem["Seed"]))
    return len(seeds_found)

def get_avg_w_ij(csvFilename, feature_names):
    data = readCSVFile(csvFilename)
    avg_w_ij = {}
    indexes = get_indexes(csvFilename)
    for index in indexes:
        avg_w_ij[index] = {}
        for feature_name in feature_names:
            avg_w_ij[index][feature_name] = 0.0

    for elem in data:
        index = int(elem["Event Index"])
        for feature_name in feature_names:
            avg_w_ij[index][feature_name] = avg_w_ij[index][feature_name] + float(elem[feature_name])

    num_seeds = get_num_seeds(csvFilename)
    for index in avg_w_ij.keys():
        for feature_name in avg_w_ij[index].keys():
            avg_w_ij[index][feature_name] = avg_w_ij[index][feature_name] / num_seeds

    return avg_w_ij

def get_avg_w_ij_abs(csvFilename, feature_names):
    data = readCSVFile(csvFilename)
    avg_w_ij = {}
    indexes = get_indexes(csvFilename)
    for index in indexes:
        avg_w_ij[index] = {}
        for feature_name in feature_names:
            avg_w_ij[index][feature_name] = 0.0

    for elem in data:
        index = int(elem["Event Index"])
        for feature_name in feature_names:
            avg_w_ij[index][feature_name] = avg_w_ij[index][feature_name] + abs(float(elem[feature_name]))

    num_seeds = get_num_seeds(csvFilename)
    for index in avg_w_ij.keys():
        for feature_name in avg_w_ij[index].keys():
            avg_w_ij[index][feature_name] = avg_w_ij[index][feature_name] / num_seeds

    return avg_w_ij

def get_average_importances(csvFilename, feature_names, avg_w_ij_func):
    i_j = {}

    avg_w_ij = avg_w_ij_func(csvFilename, feature_names)
    for feature_name in feature_names:
        i_j[feature_name] = 0.0
        for index in avg_w_ij.keys():
            i_j[feature_name] = i_j[feature_name] + abs(avg_w_ij[index][feature_name])
        i_j[feature_name] = math.sqrt(i_j[feature_name])

    return i_j

def get_event_feature_importances(csvFilename, index, feature_names):
    avg_i = {}
    for feature_name in feature_names:
        avg_i[feature_name] = 0.0

    num_found = 0

    data = readCSVFile(csvFilename)
    for elem in data:
        if int(elem["Event Index"]) == index:
            for feature_name in feature_names:
                avg_i[feature_name] = avg_i[feature_name] + float(elem[feature_name])

            num_found = num_found + 1

    for feature_name in feature_names:
        avg_i[feature_name] = avg_i[feature_name] / num_found

    print("ID: {}. FOUND: {}. AVG_I: {}".format(index, num_found, avg_i))
    return avg_i

def get_event_feature_importances_abs(csvFilename, index, feature_names):
    avg_i = {}
    for feature_name in feature_names:
        avg_i[feature_name] = 0.0

    num_found = 0

    data = readCSVFile(csvFilename)
    for elem in data:
        if int(elem["Event Index"]) == index:
            for feature_name in feature_names:
                avg_i[feature_name] = avg_i[feature_name] + abs(float(elem[feature_name]))

            num_found = num_found + 1

    for feature_name in feature_names:
        avg_i[feature_name] = avg_i[feature_name] / num_found

    print("ID: {}. FOUND: {}. AVG_I: {}".format(index, num_found, avg_i))
    return avg_i

def cleanup_feature_names(elems):
    cleanup_lookup = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "donki_ha": "Half Width",
        "donki_speed": "Linear Speed",
        "Accel": "Acceleration",
        "2nd_order_speed_final": "2nd order speed final",
        "2nd_order_speed_20R": "2nd order speed at 20 solar radii",
        "Central_PA": "CPA",
        "MPA": "MPA",
        "halo": "Halo",
        "CMEs_past_month": "CMEs in past month",
        "CMEs_past_9_hours": "CMEs in past 9 hours",
        "CMEs_over_1000_past_9_hrs": "CMEs over 1000 km/s past 9 hrs",
        "Max_speed_past_day": "Max speed past day",
        "V log V": "V Log V",
        "richardson_formula_degrees_phi_2_solar_wind": "Richardson's equation",
        "diffusive_shock": "Diffusive shock",
        "sunspots": "Daily Sunspot Count",
        "Type_2_Area": "Type II Visualization Area",
    }
    cleaned_elems = [x for x in elems]
    for elem in cleaned_elems:
        elem[0] = cleanup_lookup[elem[0]]
    return cleaned_elems

def extract_groups(cleaned_elems):
    speed_group_features = [
        "Linear Speed",
        "Diffusive shock",
        "2nd order speed final",
        "2nd order speed at 20 solar radii",
        "V Log V",
    ]
    location_group_features = [
        "Latitude",
        "Longitude",
        "Richardson's equation",
        "CPA",
        "MPA",
    ]
    size_group_features = [
        "CPA",
        "Halo",
        "Half Width",
    ]
    history_group_features = [
        "Max speed past day",
        "CMEs in past month",
        "CMEs in past 9 hours",
        "CMEs over 1000 km/s past 9 hrs",
    ]
    other_group_features = [
        "Acceleration",
        "Type II Visualization Area",
        "Daily Sunspot Count",
    ]
    groups = {
        "Speed": 0.0,
        "Location": 0.0,
        "CME History": 0.0,
        "Other": 0.0,
        "Size": 0.0
    }

    for elem in cleaned_elems:
        if elem[0] in speed_group_features:
            groups["Speed"] = groups["Speed"] + abs(elem[1])
        if elem[0] in location_group_features:
            if elem[0] == "CPA":
                groups["Location"] = groups["Location"] + 0.5 * abs(elem[1])
            else:
                groups["Location"] = groups["Location"] + abs(elem[1])
        if elem[0] in size_group_features:
            if elem[0] == "CPA":
                groups["Size"] = groups["Size"] + 0.5 * abs(elem[1])
            else:
                groups["Size"] = groups["Size"] + abs(elem[1])
        if elem[0] in history_group_features:
            groups["CME History"] = groups["CME History"] + abs(elem[1])
        if elem[0] in other_group_features:
            groups["Other"] = groups["Other"] + abs(elem[1])

    return groups

def normalize_groups(groups):
    group_sum = 0.0
    for group in groups:
        group_sum = group_sum + group[1]
    normalized_groups = [[x[0], x[1] / group_sum] for x in groups]
    return normalized_groups

def get_contribution(feature_names, w_ij, x_ij):
    x_index = 0
    aug_w_ij = {}
    for feature_name in feature_names:
        found_index = -1
        for i in range(0, len(w_ij)):
            elem = w_ij[i]
            if feature_name in elem[0]:
                found_index = i
                break
        if found_index == -1:
            print("ERROR: CANNOT FIND " + feature_name)

        l = []
        l.extend(w_ij[found_index])
        l.append(w_ij[found_index][1] * x_ij[x_index])
        x_index = x_index + 1
        aug_w_ij[feature_name] = l

    print("\n\nW_ij:\n\n", w_ij, "\n\nX_ij\n\n", x_ij, "\n\n")

    mod_w_ij = []
    for elem in w_ij:
        feature_name = elem[0]
        mod_w_ij.append(aug_w_ij[feature_name])

    return mod_w_ij

def get_contribution_for_event(csvFilename, event_x_map, event_index, feature_names):
    full_details_data = readCSVFile(csvFilename)
    collected_events = []
    for elem in full_details_data:
        if int(elem["Event Index"]) == event_index:
            collected_events.append(elem)

    results = []
    for event in collected_events:
        m = {}
        for feature_name in feature_names:
            encoded_val = event[feature_name]
            encoded_val = encoded_val.replace("'", "")
            encoded_val = encoded_val.replace("(", "")
            encoded_val = encoded_val.replace(")", "")
            vals = encoded_val.split(',')
            feature_and_relation = vals[0].strip()
            w_ij = float(vals[1].strip())
            feature_value = event_x_map[feature_name]

            feature_name_index = feature_and_relation.index(feature_name)

            is_feature_valid = True
            relation_before = feature_and_relation[0:feature_name_index]
            relation_after = feature_and_relation[feature_name_index + len(feature_name):]

            less_than_index = relation_before.find('<')
            equal_index = relation_before.find('=')
            greater_than_index = relation_before.find('>')
            if less_than_index != -1:
                relation_value = float(relation_before[0:less_than_index].strip())
            elif greater_than_index != -1:
                relation_value = float(relation_before[0:greater_than_index].strip())
            else:
                relation_value = ''

            # On left hand side, < means > and > means <
            if relation_value != '':
                if less_than_index != -1:
                    if equal_index != -1:
                        is_feature_valid = is_feature_valid and feature_value >= relation_value
                    else:
                        is_feature_valid = is_feature_valid and feature_value > relation_value
                if greater_than_index != -1:
                    if equal_index != -1:
                        is_feature_valid = is_feature_valid and feature_value <= relation_value
                    else:
                        is_feature_valid = is_feature_valid and feature_value < relation_value

                if less_than_index == -1 and greater_than_index == -1 and equal_index != -1:
                    is_feature_valid = is_feature_valid and feature_value == relation_value

            less_than_index = relation_after.find('<')
            equal_index = relation_after.find('=')
            greater_than_index = relation_after.find('>')
            if less_than_index != -1:
                if equal_index != -1:
                    relation_value = float(relation_after[less_than_index+2:].strip())
                else:
                    relation_value = float(relation_after[less_than_index+1:].strip())
            elif greater_than_index != -1:
                if equal_index != -1:
                    relation_value = float(relation_after[greater_than_index+1:].strip())
                else:
                    relation_value = float(relation_after[greater_than_index+2:].strip())
            else:
                relation_value = ''

            if relation_value != '':
                if less_than_index != -1:
                    if equal_index != -1:
                        is_feature_valid = is_feature_valid and feature_value <= relation_value
                    else:
                        is_feature_valid = is_feature_valid and feature_value < relation_value
                if greater_than_index != -1:
                    if equal_index != -1:
                        is_feature_valid = is_feature_valid and feature_value >= relation_value
                    else:
                        is_feature_valid = is_feature_valid and feature_value > relation_value

                if less_than_index == -1 and greater_than_index == -1 and equal_index != -1:
                    is_feature_valid = is_feature_valid and feature_value == relation_value

            if is_feature_valid:
                m[feature_name] = [1, w_ij]
            else:
                m[feature_name] = [0, w_ij]
                print("\n\n{}: ZERO\n\n".format(feature_name))

        results.append(m)

    return results

            

# Overall feature importance values normalized for crt+AE 70
crt_ae_70_overall_elems = [
    ["V log V", 0.116951246],
    ["diffusive_shock", 0.111314661],
    ["donki_speed", 0.08645347],
    ["richardson_formula_degrees_phi_2_solar_wind", 0.084136939],
    ["2nd_order_speed_final", 0.071070574],
    ["2nd_order_speed_20R", 0.066563763],
    ["Type_2_Area", 0.055894284],
    ["CMEs_over_1000_past_9_hrs", 0.052178789],
    ["Max_speed_past_day", 0.049068364],
    ["halo", 0.047159094],
    ["CMEs_past_month", 0.042612146],
    ["longitude", 0.040004628],
    ["MPA", 0.036664914],
    ["sunspots", 0.029103887],
    ["donki_ha", 0.0289944],
    ["latitude", 0.027736322],
    ["Accel", 0.025557936],
    ["Central_PA", 0.014376369],
    ["CMEs_past_9_hours", 0.014158214],
] 

# Overall feature importance values normalized for dl+rRT+AE 0.6
dl_rrt_ae_0_6_overall_elems = [
    ["V log V", 0.127714828],
    ["diffusive_shock", 0.125075156],
    ["donki_speed", 0.093319164],
    ["richardson_formula_degrees_phi_2_solar_wind", 0.082312026],
    ["2nd_order_speed_20R", 0.074882766],
    ["2nd_order_speed_final", 0.07487371],
    ["CMEs_over_1000_past_9_hrs", 0.052938608],
    ["Max_speed_past_day", 0.050982241],
    ["Type_2_Area", 0.048566585],
    ["longitude", 0.036418032],
    ["CMEs_past_month", 0.033770111],
    ["donki_ha", 0.031464244],
    ["sunspots", 0.030057458],
    ["MPA", 0.028036886],
    ["Accel", 0.027213699],
    ["latitude", 0.026861452],
    ["CMEs_past_9_hours", 0.023446559],
    ["halo", 0.020301929],
    ["Central_PA", 0.011764546],
]

def pretty_print(overall_elems):   

    cleaned_elems = cleanup_feature_names(overall_elems)

    for i in range(0, len(cleaned_elems)):
        elem = cleaned_elems[i]
        print("{}. {} & {:.3f}\\\\".format(i+1, elem[0], elem[1]))

    print("\n\n\n")
    groups = extract_groups(cleaned_elems)
    sorted_groups = list(sorted(groups.items(), reverse=True, key=lambda item: item[1]))
    print("Overall Groups: \n", sorted_groups)
    print("Group & Combined Importance\\\\")
    for i in range(0, len(sorted_groups)):
        print("{}. {} & {:.3f}\\\\".format(i+1, sorted_groups[i][0], sorted_groups[i][1]))

# W_ij for FP in cRT+AE 70
crt_ae_70_fp_elems = [
    ["V log V", 0.03314594],
    ["donki_speed", 0.027895583],
    ["2nd_order_speed_final", 0.026325132],
    ["Type_2_Area", 0.025548753],
    ["halo", 0.021609284],
    ["2nd_order_speed_20R", 0.019101193],
    ["richardson_formula_degrees_phi_2_solar_wind", 0.015464271],
    ["Central_PA", 0.007605103],
    ["MPA", 0.007471092],
    ["donki_ha", 0.007303794],
    ["sunspots", 0.005812999],
    ["CMEs_past_9_hours", 0.005547225],
    ["diffusive_shock", 0.002906716],
    ["Max_speed_past_day", 0.001936058],
    ["latitude", 0.001894678],
    ["CMEs_over_1000_past_9_hrs", 0.000662097],
    ["longitude", 0.00012593],
    ["CMEs_past_month", -0.007416279],
    ["Accel", -0.012856174],
]

# W_ij for FN in cRT+AE 70
crt_ae_70_fn_elems = [
    ["V log V", 0.011861014],
    ["halo", 0.010077319],
    ["donki_speed", 0.007748308],
    ["Type_2_Area", 0.006888404],
    ["2nd_order_speed_final", 0.006302657],
    ["richardson_formula_degrees_phi_2_solar_wind", 0.006177354],
    ["sunspots", 0.002407899],
    ["MPA", 0.001949921],
    ["CMEs_over_1000_past_9_hrs", 0.001904436],
    ["Central_PA", 0.001640205],
    ["donki_ha", 0.000889095],
    ["longitude", -0.000133972],
    ["2nd_order_speed_20R", -0.000428688],
    ["CMEs_past_month", -0.000545776],
    ["CMEs_past_9_hours", -0.000639404],
    ["Accel", -0.000695779],
    ["latitude", -0.000748759],
    ["Max_speed_past_day", -0.002193298],
    ["diffusive_shock", -0.006038405],
]

# W_ij for FP in dl+rRT+AE 0.6
dl_rrt_ae_0_6_fp_elems = [
    ["V log V", 0.13958964],
    ["donki_speed", 0.104527426],
    ["2nd_order_speed_final", 0.091431498],
    ["Type_2_Area", 0.080354815],
    ["halo", 0.063943725],
    ["richardson_formula_degrees_phi_2_solar_wind", 0.06273632],
    ["2nd_order_speed_20R", 0.041282848],
    ["MPA", 0.022937079],
    ["Central_PA", 0.022554993],
    ["sunspots", 0.019630205],
    ["donki_ha", 0.019065672],
    ["CMEs_past_9_hours", 0.017622861],
    ["CMEs_over_1000_past_9_hrs", 0.008794393],
    ["latitude", 0.003751447],
    ["Max_speed_past_day", -0.000999224],
    ["longitude", -0.002820117],
    ["CMEs_past_month", -0.024508687],
    ["diffusive_shock", -0.032204463],
    ["Accel", -0.039624104],
]

# W_ij for FN in dl+rRT+AE 0.6
dl_rrt_ae_0_6_fn_elems = [
    ["V log V", 0.088014908],
    ["donki_speed", 0.0527415],
    ["2nd_order_speed_final", 0.038528083],
    ["halo", 0.038141255],
    ["richardson_formula_degrees_phi_2_solar_wind", 0.037522062],
    ["Type_2_Area", 0.031263681],
    ["CMEs_over_1000_past_9_hrs", 0.013712516],
    ["sunspots", 0.011963599],
    ["MPA", 0.00808115],
    ["Central_PA", 0.007075343],
    ["donki_ha", 0.001220516],
    ["CMEs_past_9_hours", -0.000623279],
    ["longitude", -0.002349981],
    ["CMEs_past_month", -0.00285836],
    ["Accel", -0.004574332],
    ["latitude", -0.004637048],
    ["Max_speed_past_day", -0.014877185],
    ["2nd_order_speed_20R", -0.015796582],
    ["diffusive_shock", -0.064054838],
]

def pretty_print_event(feature_names, fp_wijs, fp_x_ij, fn_wijs, fn_x_ij, overall_i_js, mul_contribution):
    fp_elems = get_contribution(feature_names, fp_wijs, fp_x_ij)
    fp_elems = cleanup_feature_names(fp_elems)

    fn_elems = get_contribution(feature_names, fn_wijs, fn_x_ij)
    fn_elems = cleanup_feature_names(fn_elems)

    overall_elems = cleanup_feature_names(overall_i_js)
    ordered_grouped = []
    for elem in overall_elems:
        l = []
        for i in range(0, len(fp_elems)):
            fp_it = fp_elems[i]
            if elem[0] in fp_it[0]:
                arr = [i+1]
                arr.extend(fp_it)
                l.append(arr)
                break
        for i in range(0, len(fn_elems)):
            fn_it = fn_elems[i]
            if elem[0] in fn_it[0]:
                arr = [i+1]
                arr.extend(fn_it)
                l.append(arr)
                break
        ordered_grouped.append(l)

    print("Feature & \\multicolumn{4}{c|}{FP} & \\multicolumn{4}{c}{FN}\\\\")

    fp_top_contributors_sorted = sorted(ordered_grouped, reverse=True, key=lambda x : x[0][3])
    for i in range(0, len(fp_top_contributors_sorted)):
        fp_feature_name = fp_top_contributors_sorted[i][0][1]
        contributor_rank = i+1
        for j in range(0, len(ordered_grouped)):
            if ordered_grouped[j][0][1] == fp_feature_name:
                ordered_grouped[j][0].append(contributor_rank)
                break

    fp_top_contributors_indexes = [
        fp_top_contributors_sorted[0][0][0],
        fp_top_contributors_sorted[1][0][0],
        fp_top_contributors_sorted[2][0][0]
    ]

    fn_top_contributors_sorted = sorted(ordered_grouped, key=lambda x : x[1][3])

    for i in range(0, len(fn_top_contributors_sorted)):
        fn_feature_name = fn_top_contributors_sorted[i][0][1]
        contributor_rank = len(fn_top_contributors_sorted) - i
        for j in range(0, len(ordered_grouped)):
            if ordered_grouped[j][1][1] == fn_feature_name:
                ordered_grouped[j][1].append(contributor_rank)
                break

    fn_top_contributors_indexes = [
        fn_top_contributors_sorted[0][1][0],
        fn_top_contributors_sorted[1][1][0],
        fn_top_contributors_sorted[2][1][0]
    ]

    e = 0.0
    p = 0.0
    n = 0.0
    for i in range(0, len(fp_top_contributors_sorted)):
        e = e + fp_top_contributors_sorted[i][0][3]
        if fp_top_contributors_sorted[i][0][3] > 0:
            p = p + fp_top_contributors_sorted[i][0][3]
        if fp_top_contributors_sorted[i][0][3] < 0:
            n = n + fp_top_contributors_sorted[i][0][3]
    print("FP E: {}. P: {}. N: {}".format(e, p, n))

    e = 0.0
    p = 0.0
    n = 0.0
    for i in range(0, len(fn_top_contributors_sorted)):
        e = e + fn_top_contributors_sorted[i][1][3]
        if fn_top_contributors_sorted[i][1][3] > 0:
            p = p + fn_top_contributors_sorted[i][1][3]
        if fn_top_contributors_sorted[i][1][3] < 0:
            n = n + fn_top_contributors_sorted[i][1][3]
    print("FN E: {}. P: {}. N: {}".format(e, p, n))
    
    for i in range(0, len(ordered_grouped)):
        elem = ordered_grouped[i]

        row = "{} & ".format(elem[0][1])
        if elem[0][0] == 1:
            row = row + "\\textbf{" + str(elem[0][0]) + "} & \\textbf{" + "{:.3f}".format(elem[0][2]) + "}"
        elif elem[0][0] == 2:
            row = row + "\\emph{" + str(elem[0][0]) + "} & \\emph{" + "{:.3f}".format(elem[0][2]) + "}"
        elif elem[0][0] == 3:
            row = row + "\\underline{" + str(elem[0][0]) + "} & \\underline{" + "{:.3f}".format(elem[0][2]) + "}"
        else:
            row = row + str(elem[0][0]) + " & {:.3f}".format(elem[0][2])

        if elem[0][0] == fp_top_contributors_indexes[0]:
            row = row + " & \\textbf{" + str(elem[0][4]) + "} & \\textbf{" + "{:.3f}".format(elem[0][3] * mul_contribution) + "}"
        elif elem[0][0] == fp_top_contributors_indexes[1]:
            row = row + " & \\emph{" + str(elem[0][4]) + "} & \\emph{" + "{:.3f}".format(elem[0][3] * mul_contribution) + "}"
        elif elem[0][0] == fp_top_contributors_indexes[2]:
            row = row + " & \\underline{" + str(elem[0][4]) + "} & \\underline{" + "{:.3f}".format(elem[0][3] * mul_contribution) + "}"
        else:
            row = row + " & {} & {:.3f}".format(elem[0][4], elem[0][3] * mul_contribution)

        row = row + " & "
        
        if elem[1][0] == 19:
            row = row + "\\textbf{" + str(elem[1][0]) + "} & \\textbf{" + "{:.3f}".format(elem[1][2]) + "}"
        elif elem[1][0] == 18:
            row = row + "\\emph{" + str(elem[1][0]) + "} & \\emph{" + "{:.3f}".format(elem[1][2]) + "}"
        elif elem[1][0] == 17:
            row = row + "\\underline{" + str(elem[1][0]) + "} & \\underline{" + "{:.3f}".format(elem[1][2]) + "}"
        else:
            row = row + str(elem[1][0]) + " & {:.3f}".format(elem[1][2])

        if elem[1][0] == fn_top_contributors_indexes[0]:
            row = row + " & \\textbf{" + str(elem[1][4]) + "} & \\textbf{" + "{:.3f}".format(elem[1][3] * mul_contribution) + "}"
        elif elem[1][0] == fn_top_contributors_indexes[1]:
            row = row + " & \\emph{" + str(elem[1][4]) + "} & \\emph{" + "{:.3f}".format(elem[1][3] * mul_contribution) + "}"
        elif elem[1][0] == fn_top_contributors_indexes[2]:
            row = row + " & \\underline{" + str(elem[1][4]) + "} & \\underline{" + "{:.3f}".format(elem[1][3] * mul_contribution) + "}"
        else:
            row = row + " & {} & {:.3f}".format(elem[1][4], elem[1][3] * mul_contribution)

        row = row + "\\\\"
        print(row)

    print("\n\n\n")

    overall_groups = extract_groups(overall_elems)
    sorted_overall_groups = list(sorted(overall_groups.items(), reverse=True, key=lambda item: item[1]))

    fp_groups = extract_groups(fp_elems)
    sorted_fp_groups = list(sorted(fp_groups.items(), reverse=True, key=lambda item: item[1]))
    sorted_fp_groups = normalize_groups(sorted_fp_groups)
    fn_groups = extract_groups(fn_elems)
    sorted_fn_groups = list(sorted(fn_groups.items(), reverse=True, key=lambda item: item[1]))
    sorted_fn_groups = normalize_groups(sorted_fn_groups)
    print("Overall GROUPS")
    print(sorted_overall_groups)
    print("FP GROUPS")
    print(sorted_fp_groups)
    print("FN GROUPS")
    print(sorted_fn_groups)

    fp_groups_rankings = []
    for i in range(0, len(sorted_fp_groups)):
        group = sorted_fp_groups[i][0]
        for j in range(0, len(sorted_fp_groups)):
            if sorted_fp_groups[j][0] == group:
                fp_groups_rankings.append(j+1)
                break

    fn_groups_rankings = []
    for i in range(0, len(sorted_fp_groups)):
        group = sorted_fp_groups[i][0]
        for j in range(0, len(sorted_fn_groups)):
            if sorted_fn_groups[j][0] == group:
                fn_groups_rankings.append(j+1)
                break

    overall_groups_rankings = []
    for i in range(0, len(sorted_fp_groups)):
        group = sorted_fp_groups[i][0]
        for j in range(0, len(sorted_overall_groups)):
            if sorted_overall_groups[j][0] == group:
                overall_groups_rankings.append(j+1)
                break

    print("Group & Overall & FP & FN\\\\")
    for i in range(0, len(sorted_fp_groups)):
        group_name = sorted_fp_groups[i][0]

        overall_group_ranking = overall_groups_rankings[i]
        overall_group_val = 0
        for j in range(0, len(sorted_overall_groups)):
            if group_name == sorted_overall_groups[j][0]:
                overall_group_val = sorted_overall_groups[j][1]
                break

        fp_group_ranking = fp_groups_rankings[i]
        fp_group_val = 0
        for j in range(0, len(sorted_fp_groups)):
            if group_name == sorted_fp_groups[j][0]:
                fp_group_val = sorted_fp_groups[j][1]
                break

        fn_group_ranking = fn_groups_rankings[i]
        fn_group_val = 0
        for j in range(0, len(sorted_fn_groups)):
            if group_name == sorted_fn_groups[j][0]:
                fn_group_val = sorted_fn_groups[j][1]
                break

        print("{} & {} {:.3f} & {} {:.3f} & {} {:.3f}\\\\".format(sorted_fp_groups[i][0], overall_group_ranking, overall_group_val, fp_group_ranking, fp_group_val, fn_group_ranking, fn_group_val))
        

feature_names = np.array([
    "donki_speed",
    "donki_ha",
    "longitude",
    "latitude",
    "Accel",
    "Type_2_Area",
    "2nd_order_speed_final",
    "2nd_order_speed_20R",
    "Central_PA",
    "MPA",
    "sunspots",
    "CMEs_past_month",
    "CMEs_past_9_hours",
    "CMEs_over_1000_past_9_hrs",
    "Max_speed_past_day",
    "richardson_formula_degrees_phi_2_solar_wind",
    "V log V",
    "halo",
    "diffusive_shock",
])

index_map, all_data, reverse_index_map, training_data = collect_data()

class_names = np.array(["SEP", "Non-SEP"])

explainer = lime.lime_tabular.LimeTabularExplainer(all_data, feature_names=feature_names, categorical_features=[17], class_names=class_names, discretize_continuous=False)

fp_event = all_data[index_map[55]]
fp_event_map = {}
for i in range(0, len(feature_names)):
    feature_name = feature_names[i]
    fp_event_map[feature_name] = fp_event[i]

fn_event = all_data[index_map[868]]
fn_event_map = {}
for i in range(0, len(feature_names)):
    feature_name = feature_names[i]
    fn_event_map[feature_name] = fn_event[i]

#iris = sklearn.datasets.load_iris()
#train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)
#rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
#rf.fit(train, labels_train)
#print(labels_test[0:5])

#diabetes = sklearn.datasets.load_diabetes()
#train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(diabetes.data, diabetes.target, train_size=0.80)
#rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
#rf.fit(train, labels_train)
#print('Random Forest MSError', np.mean((rf.predict(test) - labels_test) ** 2))
#print('MSError when predicting the mean', np.mean((labels_train.mean() - labels_test) ** 2))
#categorical_features = np.argwhere(np.array([len(set(diabetes.data[:,x])) for x in range(diabetes.data.shape[1])]) <= 173).flatten()
#explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=diabetes.feature_names, class_names=['disease'], categorical_features=categorical_features, verbose=True, mode='regression')
#i = 25
#exp = explainer.explain_instance(test[i], rf.predict, num_features=len(diabetes.feature_names))
#exp.save_to_file("../eval/testing.html")


#header_names, all_rows = get_all_metrics_crt(training_data, all_data, reverse_index_map, feature_names, class_names, get_all_metrics_for_seed)
#output_metrics("../eval/classification_lime_metrics.csv", header_names, all_rows)

#header_names, all_rows = get_all_metrics_dl(training_data, all_data, reverse_index_map, feature_names, get_all_metrics_for_seed)
#output_metrics("../eval/dl_lime_metrics.csv", header_names, all_rows)

#avg_i_j = get_average_importances("../eval/classification_lime_metrics.csv", feature_names, get_avg_w_ij)
#print(avg_i_j)

#get_event_feature_importances("../eval/classification_lime_metrics.csv", 55, feature_names)
#get_event_feature_importances("../eval/classification_lime_metrics.csv", 868, feature_names)

#pretty_print(crt_ae_70_overall_elems)

#pretty_print_event(feature_names, crt_ae_70_fp_elems, fp_event, crt_ae_70_fn_elems, fn_event, crt_ae_70_overall_elems, 1e3)

#fp_contribution = get_contribution_for_event("../eval/classification_lime_metrics_full_details.csv", fp_event_map, 55, feature_names)
#print(fp_contribution)

#fn_contribution = get_contribution_for_event("../eval/classification_lime_metrics_full_details.csv", fn_event_map, 868, feature_names)
#print(fn_contribution)


#avg_i_j = get_average_importances("../eval/dl_lime_metrics.csv", feature_names, get_avg_w_ij)
#print(avg_i_j)

#get_event_feature_importances("../eval/dl_lime_metrics.csv", 55, feature_names)
#get_event_feature_importances("../eval/dl_lime_metrics.csv", 868, feature_names)

#pretty_print(dl_rrt_ae_0_6_overall_elems)

pretty_print_event(feature_names, dl_rrt_ae_0_6_fp_elems, fp_event, dl_rrt_ae_0_6_fn_elems, fn_event, dl_rrt_ae_0_6_overall_elems, 1e3)

#predicted_vals = {}
#arr = crt_ae_70_predict_proba(all_data)
#for i in range(0, len(arr)):
#    predicted_vals[reverse_index_map[i]] = arr[i][1]

#with open("../eval/crt_ae_predictions_all.csv", 'w', newline='') as csvfile:

#    fieldnames = ["dummy", "index", "donki_date", "cdaw_date", "donki_speed", "donki_ha", "longitude", "latitude", "Accel", "2nd_order_speed_final", "2nd_order_speed_20R", "Central_PA", "MPA", "sunspots", "halo", "target", "100MeV_peak_intensity", "100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln", "threshold_time", "peak_time", "expected_richardson", "expected_richardson_ln", "Type_2_Area", "richardson_formula_degrees_phi_2_solar_wind", "diffusive_shock", "V log V", "CMEs_past_month", "CMEs_past_9_hours", "CMEs_over_1000_past_9_hrs", "Max_speed_past_day", "solar_wind_speed", "connection_angle_degrees", "connection_angle_degrees_phi_2_solar_wind_sq_div", "trained_richardson_ln", "2nd_order_speed_initial", "donki_speed_unnormalized", "Double_CME_100_MeV"]
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    writer.writeheader()

#    train_data = readCSVFile("../res/gen/firstStageAllTraining.csv")
#    test_data = readCSVFile("../res/gen/firstStageTest.csv")
#    for elem in train_data + test_data:
#        row = {}
#        for key in elem.keys():
#            if "dummy" in key:
#                row["dummy"] = elem[key]
#            else:
#                row[key] = elem[key]
#        row["predicted_100MeV_peak_intensity_ln"] = predicted_vals[int(elem["index"])]
#        writer.writerow(row)


