import csv
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

def estimateParameters(data):
    peak_intensities = []
    for elem in data:
        peak_intensity = float(elem["100MeV_peak_intensity"])
        peak_intensities.append(peak_intensity)

    peak_intensities.sort()
    non_dup_index = -1
    val = peak_intensities[0]
    for i in range(1, len(peak_intensities)):
        if not math.isclose(peak_intensities[i], val):
            non_dup_index = i
            break

    #if non_dup_index != -1:
    #    peak_intensities = peak_intensities[non_dup_index - 1:]

    #dup_background_times = 75
    #for i in range(0, dup_background_times):
    #    peak_intensities.insert(0, peak_intensities[0])

    ln_peak_intensities = []
    sum_ln_peak_intensities = 0
    for peak_intensity in peak_intensities:
        ln_peak_intensity = math.log(peak_intensity)
        ln_peak_intensities.append(ln_peak_intensity)
        sum_ln_peak_intensities = sum_ln_peak_intensities + ln_peak_intensity

    n = len(peak_intensities)
    m = min(peak_intensities)
    alpha = float(n) / (sum_ln_peak_intensities - float(n) * math.log(m))
    mle = n * math.log(alpha) + n * alpha * math.log(m) - (alpha + 1) * sum_ln_peak_intensities
    return (peak_intensities, n, m, alpha, mle)

def estimateParametersAlt(data):
    peak_intensities = []
    for elem in data:
        peak_intensity = float(elem["100MeV_peak_intensity"])
        peak_intensities.append(peak_intensity)

    #if non_dup_index != -1:
    #    peak_intensities = peak_intensities[non_dup_index - 1:]

    #dup_background_times = 75
    #for i in range(0, dup_background_times):
    #    peak_intensities.insert(0, peak_intensities[0])

    # c = min_peak_intensity
    # alpha_est = n / (sum_{i=1}^{n}{X_i / c})
    # https://www.casact.org/sites/default/files/database/astin_vol20no2_201.pdf
    ln_peak_intensities = []
    for peak_intensity in peak_intensities:
        ln_peak_intensities.append(math.log(peak_intensity))
    m = min(ln_peak_intensities)

    # Move all values up so that the background events have a value of 1
    mod_ln_peak_intensities = []
    for ln_peak_intensity in ln_peak_intensities:
        mod_ln_peak_intensities.append(ln_peak_intensity - m + 1)
    m = min(mod_ln_peak_intensities)

    sum_ln_peak_intensities = 0
    for peak_intensity in mod_ln_peak_intensities:
        sum_ln_peak_intensities = sum_ln_peak_intensities + math.log(peak_intensity / m)

    n = len(mod_ln_peak_intensities)
    alpha = float(n) / sum_ln_peak_intensities
    return (peak_intensities, mod_ln_peak_intensities, n, m, alpha)

def getPareto(peak_intensities, m, alpha):
    pareto_est = []
    for elem in peak_intensities:
        pareto_est.append((alpha * math.pow(float(m), alpha)) / (math.pow(elem, alpha + 1)))
    return pareto_est

def get_p_prime(p_y):
    # p'(y) = [p(y) - min(p(Y))] / [max(p(Y)) - min(p(Y))]
    min_p_y = min(p_y)
    max_p_y = max(p_y)
    p_prime_y = []
    for elem in p_y:
        val = (elem - min_p_y) / float((max_p_y - min_p_y))
        p_prime_y.append(val)
    return p_prime_y

def get_f_w_prime(p_prime_y, f_w_prime_alpha):
    # f_w'(alpha, y) = 1 - alpha * p'(y)
    f_w_prime = []
    for elem in p_prime_y:
        val = 1 - f_w_prime_alpha * elem
        f_w_prime.append(val)
    return f_w_prime

def get_f_w_prime_sq(f_w_prime, epsilon):
    # f_w''(alpha, y) = max(1 - alpha * p'(y), epsilon)
    f_w_prime_sq = []
    for elem in f_w_prime:
        val = max((elem, epsilon))
        f_w_prime_sq.append(val)
    return f_w_prime_sq

def get_f_w(f_w_prime_sq):
    s = 0
    for elem in f_w_prime_sq:
        s = s + elem
    denom = float(s) / len(f_w_prime_sq)

    print(denom)

    f_w = []
    for elem in f_w_prime_sq:
        val = elem / denom
        f_w.append(val)
    return f_w

def apply_feature_to_data(data, feature_values, feature_name):
    for i in range(0, len(data)):
        data[i][feature_name] = feature_values[i]
    return

def main():
    #data = readCSVFile("../res/adapted_rRT_data_learn_richardson.csv")
    data = readCSVFile("../res/gen/firstStageAllTraining.csv")
    peak_intensities, mod_ln_peak_intensities, n, m, alpha = estimateParametersAlt(data)
    for i in range(0, 21):
        f_w_prime_alpha = 0.1 * i

        print("ALPHA: {}".format(f_w_prime_alpha))

        p_y = getPareto(mod_ln_peak_intensities, m, alpha)
        p_prime_y = get_p_prime(p_y)
        f_w_prime = get_f_w_prime(p_prime_y, f_w_prime_alpha)
        f_w_prime_sq = get_f_w_prime_sq(f_w_prime, epsilon = 1e-3)
        f_w = get_f_w(f_w_prime_sq)

        feature_name = "dense_loss_weighting_func_alpha_{:.2f}".format(f_w_prime_alpha)
        apply_feature_to_data(data, f_w, feature_name)

    p_y = getPareto(mod_ln_peak_intensities, m, alpha)
    feature_name = "p_y"
    apply_feature_to_data(data, p_y, feature_name)

    writeToCSV(data, "features_with_dense_loss.csv")

if __name__ == "__main__":
    main()
