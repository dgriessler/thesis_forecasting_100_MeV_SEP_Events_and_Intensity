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

def calculate_connection_angle(data, newFeatureName):
    angular_speed_of_sun = 360 / (27.27 * 86400)
    au = 1.5 * math.pow(10, 8)
    for elem in data:
        # connection angle = arccos(sin(theta_1) * sin(theta_2) + cos(theta_1) * cos(theta_2) * cos(phi_1 - phi_2))
        # theta_1 = latitude (in degrees)
        # phi_1 = longitude (in degrees)
        # theta_2 = 0
        # phi_2 = angular_speed_of_sun * 1 AU / solar_wind_speed (in degrees)
        # angular_speed_of_sun = 360 / 27.27 * 86400) degrees/second
        # 1 AU = 1.5 * 10^8 km
        theta_1 = elem["latitude"]
        phi_1 = elem["longitude"]
        theta_2 = 0
        phi_2 = angular_speed_of_sun * au / elem["solar_wind_speed"]
        # math.acos wants RADIANS not DEGREES
        # math.sin wants RADIANS not DEGREES
        # math.cos wants RADIANS not DEGREES
        theta_1_rad = math.radians(theta_1)
        phi_1_rad = math.radians(phi_1)
        theta_2_rad = math.radians(theta_2)
        phi_2_rad = math.radians(phi_2)
        # math.acos only returns positive values. The connection angle sometimes should be negative
        if phi_1 >= phi_2 - 180 and phi_1 <= phi_2:
            sign = -1
        else:
            sign = 1
        
        # math.acos returns RADIANS but we want DEGREES
        connection_angle_rad = sign * math.acos(math.sin(theta_1_rad) * math.sin(theta_2_rad) + math.cos(theta_1_rad) * math.cos(theta_2_rad) * math.cos(phi_1_rad - phi_2_rad))

        elem[newFeatureName] = math.degrees(connection_angle_rad)

def calculate_expected_richardson(data, connection_angle_feature_name, newFeatureName):
    # I = 0.013 * EXP(0.0036 * V - connection_angle^2 / (2 * sigma^2))
    # sigma = 43
    for elem in data:
        V = elem["donki_speed"]
        connection_angle = elem[connection_angle_feature_name]
        richardson_intensity = 0.013 * math.exp(0.0036 * V - math.pow(connection_angle, 2) / (2 * math.pow(43, 2)))
        elem[newFeatureName] = richardson_intensity

def calculate_expected_richardson_ln(data, connection_angle_feature_name, ricahrdson_feature_name, newFeatureName):
    # I = 0.013 * EXP(0.0036 * V - connection_angle^2 / (2 * sigma^2))
    # LN(I) = LN(0.013 * EXP(0.0036 * V - connection_angle^2 / (2 * sigma^2)))
    # LN(I) = LN(0.013) + LN(EXP(0.0036 * V - connection_angle^2 / (2 * sigma^2)))
    # LN(I) = LN(0.013) + 0.0036 * V - connection_angle^2 / (2 * sigma^2)
    # Or, just LN(I) because we already calculated the richardson feature (THEY SHOULD BE EQUAL)
    i = 0
    for elem in data:
        V = elem["donki_speed"]
        connection_angle = elem[connection_angle_feature_name]
        richardson_intensity_ln = math.log(0.013) + 0.0036 * V - math.pow(connection_angle, 2) / (2 * math.pow(43, 2))
        richardson_feature = elem[ricahrdson_feature_name]
        richardson_feature_ln = math.log(richardson_feature)
        if not math.isclose(richardson_intensity_ln, richardson_feature_ln):
            print("EXPECTED RICHARDSON_LN FEATURES TO BE CLOSE: index: {}".format(i), file=sys.stderr)

        i = i + 1
        elem[newFeatureName] = richardson_feature_ln

def parseDateTime(val):
    date_time = val.split(' ')
    month_day_year = date_time[0].split('/')
    hour_minute_second = date_time[1].split(':')
    if len(hour_minute_second) < 3:
        second = 0
    else:
        second = hour_minute_second[2]
    date_data = {
        "month": int(month_day_year[0]),
        "day" : int(month_day_year[1]),
        "year" : int(month_day_year[2]),
        "hour" : int(hour_minute_second[0]),
        "minute" : int(hour_minute_second[1]),
        "second" : int(second)
    }
    return Time(date_data, scale="utc")

def parseType2Data(type_2_data):
    parsed_data = []
    for elem in type_2_data:
        cme_date_time_str = elem["CME_Date_Time"]
        max_freq_str = elem["Max_Frequency"]
        if cme_date_time_str != "----" and max_freq_str != "????":
            start_time = parseDateTime(elem["DH_Type_II_Start"])
            end_time = parseDateTime(elem["DH_Type_II_End"])
            max_freq = float(max_freq_str)
            min_freq = float(elem["Min_Frequency"])
            cme_time = parseDateTime(cme_date_time_str)
            parsed_data.append({
                    "DH_Type_II_Start" : start_time,
                    "DH_Type_II_End" : end_time,
                    "Max_Frequency" : max_freq,
                    "Min_Frequency" : min_freq,
                    "CME_Date_Time" : cme_time
                })
    return parsed_data

def calculate_type_2_area(obj):
    tEnd = obj["DH_Type_II_End"]
    tStart = obj["DH_Type_II_Start"]
    tDiff = tEnd - tStart
    tDiffMin = (tDiff.sec) / 60.0
    maxFreq = obj["Max_Frequency"]
    minFreq = obj["Min_Frequency"]
    freqDiff = maxFreq - minFreq
    return tDiffMin * freqDiff

def apply_type_2_radio_patch(data, type_2_data, type2FeatureName):
    # For event 7/23/2012 2:36 CDAW time
    #   change: type 2 area end time to be 06:00 hours
    cdaw_date = parseDateTime("7/23/2012 02:36:00")
    data_match = None
    for elem in data:
        if parseDateTime(elem["cdaw_date"]) == cdaw_date:
            data_match = elem
            break
    type_2_match = None
    for elem in type_2_data:
        if cdaw_date == elem["CME_Date_Time"]:
                type_2_match = elem
                break

    if data_match is None:
        print("CAN'T APPLY PATCH. CDAW NOT FOUND: {}".format(cdaw_date.ymdhms), file=sys.stderr)
    elif type_2_match is None:
        print("CAN'T APPLY PATCH. CDAW TYPE 2 DATA NOT FOUND: {}".format(cdaw_date.ymdhms), file=sys.stderr)
    else:
        print("Before patch: {}".format(data_match[type2FeatureName]))
        year, month, day, hour, minute, second = type_2_match["DH_Type_II_End"].ymdhms
        modEndStr = str(month) + "/" + str(day) + "/" + str(year) + " " + str(6) + ":" + str(0) + ":" + str(0)
        obj = {
            "DH_Type_II_Start" : type_2_match["DH_Type_II_Start"],
            "DH_Type_II_End" : parseDateTime(modEndStr),
            "Max_Frequency" : type_2_match["Max_Frequency"],
            "Min_Frequency" : type_2_match["Min_Frequency"]
        }
        data_match[type2FeatureName] = calculate_type_2_area(obj)
        print("Patch applied: {}".format(data_match[type2FeatureName]))

def calculate_type_2_radio(data, newFeatureName):
    # Type 2 Area = (End time - Start time) * (Max frequency - Min frequency)
    type_2_data_raw = readCSVFile("../res/Wind_WAVES_type_II_bursts_and_CMEs.csv")
    type_2_data = parseType2Data(type_2_data_raw)

    for elem in data:
        cdaw_date = parseDateTime(elem["cdaw_date"])
        match = None
        for type_2_elem in type_2_data:
            #if cdaw_date >= type_2_elem["DH_Type_II_Start"] and cdaw_date <= type_2_elem["DH_Type_II_End"]:
            #    match = type_2_elem
            #    break
            if cdaw_date == type_2_elem["CME_Date_Time"]:
                match = type_2_elem
                break
        if match is not None:
            type_2_area = calculate_type_2_area(match)
            elem[newFeatureName] = type_2_area
        else:
            elem[newFeatureName] = 0

    apply_type_2_radio_patch(data, type_2_data, newFeatureName)

def calculate_100_MeV_ln(data, newFeatureName):
    # 100 MeV LN = LN(100 MeV Intensity)
    for elem in data:
        elem[newFeatureName] = math.log(float(elem["100MeV_peak_intensity"]))

def calculate_richardson_feature(data, newFeatureName):
    # Normal Richardson: I = 0.013 * EXP(0.0036 * V - connection_angle^2 / (2 * sigma^2))
    # Our feature version: I = 0.013 * EXP(-connection_angle^2 / (2 * sigma^2))
    # Expect connection_angle to be DEGREES
    # sigma = 43
    for elem in data:
        connection_angle = elem["connection_angle_degrees"]
        richardson_feature = 0.013 * math.exp(-1 * math.pow(connection_angle, 2) / (2 * math.pow(43, 2)))
        elem[newFeatureName] = richardson_feature

def calculate_diffusive_shock_v():
    # v (Particle speed for 100 MeV protons): (3 * 10^5 km/s) * sqrt(1 - (1/((100 MeV + 938 MeV) / 938 MeV))^2)
    v_c = 3 * math.pow(10, 5) # km/s
    v_gamma = (100.0 + 938.0) / (938.0)
    return v_c * math.sqrt(1 - math.pow((1.0 / v_gamma), 2) )

def readCdawCatalogAndExtractLinearSpeed(data, newFeatureName):
    cdawCatalog = readCSVFile("../res/internet/cdaw_univ_all.csv")

    mappedCdawCatalog = {}
    for elem in cdawCatalog:
        date_time = elem["Date_Time"]
        if date_time in mappedCdawCatalog:
            mappedCdawCatalog[date_time].append(elem)
        else:
            mappedCdawCatalog[date_time] = [elem]

    for elem in data:
        cdaw_date = elem["cdaw_date"]

        l = mappedCdawCatalog[cdaw_date]
        if len(l) == 0:
            match = None
        elif len(l) == 1:
            match = l[0]
        else:
            # More than one event is a possible match
            second_order_final = int(elem["2nd_order_speed_final"])
            second_order_initial = int(elem["2nd_order_speed_initial"])
            match = None
            for catalog_elem in l:
                catalog_elem_second_order_final = int(catalog_elem["2nd_order_speed_final"])
                catalog_elem_second_order_initial = int(catalog_elem["2nd_order_speed_initial"])
                if catalog_elem_second_order_final == second_order_final and second_order_initial == catalog_elem_second_order_initial:
                    match = catalog_elem
                    break

        if match is None:
            print("NO MATCH FOUND IN CDAW CATALOG: {}".format(cdaw_date.ymdhms), file=sys.stderr)
            break
        else:
            elem[newFeatureName] = match["Linear_Speed"]

def calculate_diffusive_shock(data, newFeatureName):
    # diffusive_shock = term_1 * term_2 * term_3 * term_4 * term_5
    # term_1 = N
    # term_2 = v
    # term_3 = 1 / (gamma - 1)
    # term_4 = 1 / (1 + term_4_sub_1)^(k+1)
    # term_4_sub_1 = Vinj^2 / (k * Vth^2)
    # term_5 = (Vinj / v)^(gamma + 1)

    # N (shock efficiency): 0.1
    # v (Particle speed for 100 MeV protons): (3 * 10^5 km/s) * sqrt(1 - (1/((100 MeV + 938 MeV) / 938 MeV))^2)
    # gamma: (if M > 1.1): (4 * M^2) / (M^2 - 1)
    # gamma: (else): (4 * 1.1^2) / (1.1^2 - 1)
    # M = Vsh / Va
    # Vsh (shock speed or Linear Speed from DONKI (not CDAW))
    # Va (Alven speed): 600 km/s
    # Vinj = 2.5 * Vsh
    # k (distribution parameter): 2
    # Vth (proton thermal speed): 150 km/s
    for elem in data:
        N = 0.1
        v = calculate_diffusive_shock_v()
        vsh = float(elem["donki_speed"])
        va = float(600)
        vinj = 2.5 * vsh
        k = float(2)
        vth = float(150)
        m = vsh / va
        if m > 1.1:
            gamma = (4 * math.pow(m, 2)) / (math.pow(m, 2) - 1)
        else:
            gamma = (4 * math.pow(1.1, 2)) / (math.pow(1.1, 2) - 1)

        term_1 = N
        term_2 = v
        term_3 = 1 / (gamma - 1)
        term_4_sub_1 = math.pow(vinj, 2) / (k * math.pow(vth, 2))
        term_4 = 1 / math.pow(1 + term_4_sub_1, k + 1)
        term_5 = math.pow(vinj / v, gamma + 1)
        diffusive_shock = term_1 * term_2 * term_3 * term_4 * term_5
        
        elem[newFeatureName] = diffusive_shock


def calculate_v_log_v(data, newFeatureName):
    for elem in data:
        v = elem["donki_speed"]
        v_log_v = v * math.log(v)
        elem[newFeatureName] = v_log_v

def sortData(e):
    return parseDateTime(e["donki_date"])

def calculate_cmes_past_month(data, newFeatureName):
    # sort by donki_date since the following algorithm depends on data being in order of donki_date
    data.sort(key=sortData)

    sec_1_min = 60
    sec_1_hour = 60 * sec_1_min
    sec_1_day = 24 * sec_1_hour
    # Just use average 30 days per month
    sec_1_month = 30 * sec_1_day

    for i in range(0, len(data)):
        current_time = parseDateTime(data[i]["donki_date"])
        cmes_past_month = 0
        for j in range(i - 1, -1, -1):
            previous_time = parseDateTime(data[j]["donki_date"])
            time_diff = (current_time - previous_time).sec
            if time_diff <= sec_1_month and current_time != previous_time:
                # Within 1 month
                cmes_past_month = cmes_past_month + 1
            else:
                # More than 1 month, so we can stop checking
                break

        data[i][newFeatureName] = cmes_past_month

def calculate_cms_past_9_hours(data, newFeatureName):
    # sort by donki_date since the following algorithm depends on data being in order of donki_date
    data.sort(key=sortData)

    sec_1_min = 60
    sec_1_hour = 60 * sec_1_min
    sec_9_hours = 9 * sec_1_hour

    for i in range(0, len(data)):
        current_time = parseDateTime(data[i]["donki_date"])
        cmes_past_9_hours = 0
        for j in range(i - 1, -1, -1):
            previous_time = parseDateTime(data[j]["donki_date"])
            time_diff = (current_time - previous_time).sec
            if time_diff <= sec_9_hours and current_time != previous_time:
                # Within 9 hours
                cmes_past_9_hours = cmes_past_9_hours + 1
            else:
                # More than 9 hours, so we can stop checking
                break

        data[i][newFeatureName] = cmes_past_9_hours

def calculate_cms_over_1000_past_9_hours(data, newFeatureName):
    # sort by donki_date since the following algorithm depends on data being in order of donki_date
    data.sort(key=sortData)

    sec_1_min = 60
    sec_1_hour = 60 * sec_1_min
    sec_9_hours = 9 * sec_1_hour
    threshold_speed = 1000

    for i in range(0, len(data)):
        current_time = parseDateTime(data[i]["donki_date"])
        cmes_over_1000_past_9_hours = 0
        for j in range(i - 1, -1, -1):
            previous_time = parseDateTime(data[j]["donki_date"])
            time_diff = (current_time - previous_time).sec
            if time_diff <= sec_9_hours and current_time != previous_time:
                # Within 9 hours
                previous_speed = int(data[j]["donki_speed"])
                if previous_speed > threshold_speed:
                    cmes_over_1000_past_9_hours = cmes_over_1000_past_9_hours + 1
            else:
                # More than 9 hours, so we can stop checking
                break

        data[i][newFeatureName] = cmes_over_1000_past_9_hours

def calculate_max_speed_past_day(data, newFeatureName):
    # sort by donki_date since the following algorithm depends on data being in order of donki_date
    data.sort(key=sortData)

    sec_1_min = 60
    sec_1_hour = 60 * sec_1_min
    sec_1_day = 24 * sec_1_hour

    for i in range(0, len(data)):
        current_time = parseDateTime(data[i]["donki_date"])
        max_speed_past_day = 0
        for j in range(i - 1, -1, -1):
            previous_time = parseDateTime(data[j]["donki_date"])
            time_diff = (current_time - previous_time).sec
            if time_diff <= sec_1_day and current_time != previous_time:
                # Within 1 day
                previous_speed = int(data[j]["donki_speed"])
                if previous_speed > max_speed_past_day:
                    max_speed_past_day = previous_speed
            else:
                # More than 1 day, so we can stop checking
                break

        data[i][newFeatureName] = max_speed_past_day

def calculate_learn_richardson_connection_angle(data, connectionAngleFeatureName, newFeatureName):
    # Learn richardson network wants donki_speed and the term with the connection_angle
    # term = -1 * connection_angle^2 / (2 * 43^2)
    for elem in data:
        connection_angle = elem[connectionAngleFeatureName]
        term = -1 * math.pow(connection_angle, 2) / (2 * math.pow(43, 2))
        elem[newFeatureName] = term

def sortDataIndex(e):
    return int(e["index"])

def main():
    data = readCSVFile("../res/features.csv")

    data = removeRows(data, feature="Double_CME_100_MeV", val=1)

    features = ["donki_speed", "longitude", "latitude", "solar_wind_speed"]
    for feature in features:
        applyFuncToFeature(data, feature, convertToFloat)

    fieldnames = list(data[0].keys())

    one_hundred_mev_intensity_ln_feature_name = "100MeV_peak_intensity_ln"
    calculate_100_MeV_ln(data, one_hundred_mev_intensity_ln_feature_name)
    fieldnames.append(one_hundred_mev_intensity_ln_feature_name)

    connection_angle_feature_name = "connection_angle_degrees"
    calculate_connection_angle(data, connection_angle_feature_name)
    fieldnames.append(connection_angle_feature_name)

    expected_richardson_feature_name = "expected_richardson"
    calculate_expected_richardson(data, connection_angle_feature_name, expected_richardson_feature_name)
    fieldnames.append(expected_richardson_feature_name)

    expected_richardson_ln_feature_name = "expected_richardson_ln"
    calculate_expected_richardson_ln(data, connection_angle_feature_name, expected_richardson_feature_name, expected_richardson_ln_feature_name)
    fieldnames.append(expected_richardson_ln_feature_name)

    type_2_feature_name = "Type_2_Area"
    calculate_type_2_radio(data, type_2_feature_name)
    fieldnames.append(type_2_feature_name)

    richardson_feature_name = "richardson_formula_degrees_phi_2_solar_wind"
    calculate_richardson_feature(data, richardson_feature_name)
    fieldnames.append(richardson_feature_name)

    diffusive_shock_feature_name = "diffusive_shock"
    calculate_diffusive_shock(data, diffusive_shock_feature_name)
    fieldnames.append(diffusive_shock_feature_name)

    v_log_v_feature_name = "V log V"
    calculate_v_log_v(data, v_log_v_feature_name)
    fieldnames.append(v_log_v_feature_name)

    cmes_past_month_feature_name = "CMEs_past_month"
    calculate_cmes_past_month(data, cmes_past_month_feature_name)
    fieldnames.append(cmes_past_month_feature_name)

    cmes_past_9_hours_feature_name = "CMEs_past_9_hours"
    calculate_cms_past_9_hours(data, cmes_past_9_hours_feature_name)
    fieldnames.append(cmes_past_9_hours_feature_name)

    cmes_over_1000_past_9_hours_feature_name = "CMEs_over_1000_past_9_hrs"
    calculate_cms_over_1000_past_9_hours(data, cmes_over_1000_past_9_hours_feature_name)
    fieldnames.append(cmes_over_1000_past_9_hours_feature_name)

    max_speed_last_day_feature_name = "Max_speed_past_day"
    calculate_max_speed_past_day(data, max_speed_last_day_feature_name)
    fieldnames.append(max_speed_last_day_feature_name)

    learn_richardson_connection_angle_feature = "connection_angle_degrees_phi_2_solar_wind_sq_div"
    calculate_learn_richardson_connection_angle(data, connection_angle_feature_name, learn_richardson_connection_angle_feature)
    fieldnames.append(learn_richardson_connection_angle_feature)

    # Resort to the provided order (by index)
    data.sort(key=sortDataIndex)
    writeCSVFile(data, "../res/adapted_rRT_data_learn_richardson.csv", fieldnames)

if __name__ == "__main__":
    main()
