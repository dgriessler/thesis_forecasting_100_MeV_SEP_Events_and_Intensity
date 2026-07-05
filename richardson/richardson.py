import csv
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def readFeatures(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def calc_richardson_ln(w_exp, w_v, V, connection_angle, sigma):
    # I = w_exp * exp(w_v * V - connection_angle^2 / (2 * (sigma)^2))
    # LN(I) = LN(w_exp) + w_v * V - connection_angle^2 / (2 * sigma^2)
    return math.log(w_exp) + w_v * V - math.pow(connection_angle, 2) / (2 * math.pow(sigma, 2))

def plotScatter(x, y, targets, plotTitle, xLabel, yLabel, pngFilename, minX, maxX, minY, maxY):
    scatterColors = []
    for i in range(0, len(targets)):
        target = targets[i]
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
        scatterColors.append(color)

    fig, ax = plt.subplots()

    scatter = ax.scatter(x, y, c=scatterColors)
    ax.set_title(plotTitle)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_xlim(minX, maxX)
    ax.set_ylim(minY, maxY)
    dottedDiagLine = mlines.Line2D([minX, maxY], [minX, maxY], color='black', ls="--")

    ax.add_line(dottedDiagLine)

    plt.savefig(pngFilename)
    plt.close(fig)

class PlotInfo:
    def __init__(self, data_x, data_y, targets, plotTitle, xLabel, yLabel, pngFilename):
        self.data_x = data_x
        self.data_y = data_y
        self.targets = targets
        self.plotTitle = plotTitle
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.pngFilename = pngFilename

    def plot(self, minX, maxX, minY, maxY):
        plotScatter(self.data_x, self.data_y, self.targets, self.plotTitle, self.xLabel, self.yLabel, self.pngFilename, minX, maxX, minY, maxY)

def range_test_true_and_learned_vals(data):
    overall_min_x = None
    overall_max_x = None
    overall_min_y = None
    overall_max_y = None
    plot_info = []

    # Original Richardson's formula
    w_exp = 0.013
    w_v = 0.0036
    sigma = 43

    data_x = []
    data_y = []
    targets = []
    for elem in data:
        V = float(elem["donki_speed"])
        connection_angle = float(elem["connection_angle_degrees"])
        data_y.append(calc_richardson_ln(w_exp, w_v, V, connection_angle, sigma))
        data_x.append(float(elem["100MeV_peak_intensity_ln"]))
        targets.append(int(elem["target"]))
        
    minX = min(data_x) - 0.1
    maxX = max(data_x) + 0.1
    minY = min(data_y) - 0.1
    maxY = max(data_y) + 0.1
    if overall_min_x is None or minX < overall_min_x:
        overall_min_x = minX
    if overall_max_x is None or maxX > overall_max_x:
        overall_max_x = maxX
    if overall_min_y is None or minY < overall_min_y:
        overall_min_y = minY
    if overall_max_y is None or maxY > overall_max_y:
        overall_max_y = maxY

    plot_info.append(PlotInfo(data_x, data_y, targets, "Original Richardson LN", "Original 100 MeV Peak Intensity LN", "Original Richardson LN", "original_richardson.png"))

    # Learned Richardson's formula
    w_exp = 0.35241104957438285
    w_v = 0.0016923099756240845
    sigma = 43

    data_x = []
    data_y = []
    targets = []
    for elem in data:
        V = float(elem["donki_speed"])
        connection_angle = float(elem["connection_angle_degrees"])
        data_y.append(calc_richardson_ln(w_exp, w_v, V, connection_angle, sigma))
        data_x.append(float(elem["100MeV_peak_intensity_ln"]))
        targets.append(int(elem["target"]))
        
    minX = min(data_x) - 0.1
    maxX = max(data_x) + 0.1
    minY = min(data_y) - 0.1
    maxY = max(data_y) + 0.1
    if overall_min_x is None or minX < overall_min_x:
        overall_min_x = minX
    if overall_max_x is None or maxX > overall_max_x:
        overall_max_x = maxX
    if overall_min_y is None or minY < overall_min_y:
        overall_min_y = minY
    if overall_max_y is None or maxY > overall_max_y:
        overall_max_y = maxY

    plot_info.append(PlotInfo(data_x, data_y, targets, "Learned Richardson LN", "Original 100 MeV Peak Intensity LN", "Learned Richardson LN", "learned_richardson.png"))

    return (overall_min_x, overall_max_x, overall_min_y, overall_max_y, plot_info)

def range_test_w_v(w_exp, data, sigma):
    # Range w_v from [0.0, 0.005, 0.001] and plot graphs

    overall_min_x = None
    overall_max_x = None
    overall_min_y = None
    overall_max_y = None
    plot_info = []

    for i in range(0, 6):
        w_v = 0.001 * i
        data_x = []
        data_y = []
        targets = []
        for elem in data:
            V = float(elem["donki_speed"])
            connection_angle = float(elem["connection_angle_degrees"])
            data_y.append(calc_richardson_ln(w_exp, w_v, V, connection_angle, sigma))
            data_x.append(float(elem["100MeV_peak_intensity_ln"]))
            targets.append(int(elem["target"]))
        
        minX = min(data_x) - 0.1
        maxX = max(data_x) + 0.1
        minY = min(data_y) - 0.1
        maxY = max(data_y) + 0.1
        if overall_min_x is None or minX < overall_min_x:
            overall_min_x = minX
        if overall_max_x is None or maxX > overall_max_x:
            overall_max_x = maxX
        if overall_min_y is None or minY < overall_min_y:
            overall_min_y = minY
        if overall_max_y is None or maxY > overall_max_y:
            overall_max_y = maxY

        plot_info.append(PlotInfo(data_x, data_y, targets, "W_V Test: {:.3f}".format(w_v), "Original 100 MeV Peak Intensity LN", "Richardson LN with w_v = {:.3f}".format(w_v), "w_v_{:.3f}.png".format(w_v)))

    return (overall_min_x, overall_max_x, overall_min_y, overall_max_y, plot_info)

def range_test_sigma(w_v, w_exp, data):
    # Range sigma [23, 63, 20] and plot graphs

    overall_min_x = None
    overall_max_x = None
    overall_min_y = None
    overall_max_y = None
    plot_info = []

    for i in range(0, 3):
        sigma = 23 + 20 * i
        data_x = []
        data_y = []
        targets = []
        for elem in data:
            V = float(elem["donki_speed"])
            connection_angle = float(elem["connection_angle_degrees"])
            data_y.append(calc_richardson_ln(w_exp, w_v, V, connection_angle, sigma))
            data_x.append(float(elem["100MeV_peak_intensity_ln"]))
            targets.append(int(elem["target"]))
        
        minX = min(data_x) - 0.1
        maxX = max(data_x) + 0.1
        minY = min(data_y) - 0.1
        maxY = max(data_y) + 0.1
        if overall_min_x is None or minX < overall_min_x:
            overall_min_x = minX
        if overall_max_x is None or maxX > overall_max_x:
            overall_max_x = maxX
        if overall_min_y is None or minY < overall_min_y:
            overall_min_y = minY
        if overall_max_y is None or maxY > overall_max_y:
            overall_max_y = maxY


        plot_info.append(PlotInfo(data_x, data_y, targets, "Sigma Test: {}".format(sigma), "Original 100 MeV Peak Intensity LN", "Richardson LN with Sigma = {}".format(sigma), "sigma_{}.png".format(sigma)))

    return (overall_min_x, overall_max_x, overall_min_y, overall_max_y, plot_info)

def range_test_w_v_and_sigma(w_exp, data):
    # Range sigma [23, 63, 20] and plot graphs
    # Range w_v from [0.0, 0.005, 0.001] and plot graphs

    overall_min_x = None
    overall_max_x = None
    overall_min_y = None
    overall_max_y = None
    plot_info = []

    for i in range(0, 3):
        sigma = 23 + 20 * i
        for j in range(0, 6):
            w_v = 0.001 * j

            data_x = []
            data_y = []
            targets = []
            for elem in data:
                V = float(elem["donki_speed"])
                connection_angle = float(elem["connection_angle_degrees"])
                data_y.append(calc_richardson_ln(w_exp, w_v, V, connection_angle, sigma))
                data_x.append(float(elem["100MeV_peak_intensity_ln"]))
                targets.append(int(elem["target"]))
        
            minX = min(data_x) - 0.1
            maxX = max(data_x) + 0.1
            minY = min(data_y) - 0.1
            maxY = max(data_y) + 0.1
            if overall_min_x is None or minX < overall_min_x:
                overall_min_x = minX
            if overall_max_x is None or maxX > overall_max_x:
                overall_max_x = maxX
            if overall_min_y is None or minY < overall_min_y:
                overall_min_y = minY
            if overall_max_y is None or maxY > overall_max_y:
                overall_max_y = maxY

            plot_info.append(PlotInfo(data_x, data_y, targets, "W_v {:.3f} Sigma {}".format(w_v, sigma), "Original 100 MeV Peak Intensity LN", "Richardson LN with w_v = {:.3f} and Sigma = {}".format(w_v, sigma), "w_v_{:.3f}_sigma_{}.png".format(w_v, sigma)))

    return (overall_min_x, overall_max_x, overall_min_y, overall_max_y, plot_info)

def main():
    # I = 0.013 * exp(0.0036 * V - connection_angle^2 / (2 * 43^2))
    # I = w_exp * exp(w_v * V - connection_angle^2 / (2 * (sigma)^2))
    w_exp = 0.013
    w_v = 0.0036
    sigma = 43

    data = readFeatures("../res/adapted_rRT_data_learn_richardson.csv")

    # Range w_v from [0.001, 0.005, 0.001] and plot graphs
    w_v_min_x, w_v_max_x, w_v_min_y, w_v_max_y, w_v_plot_info = range_test_w_v(w_exp, data, sigma)

    # Range sigma [23, 63, 20] and plot graphs
    sigma_min_x, sigma_max_x, sigma_min_y, sigma_max_y, sigma_plot_info = range_test_sigma(w_v, w_exp, data)

    # Alter both
    both_min_x, both_max_x, both_min_y, both_max_y, both_plot_info = range_test_w_v_and_sigma(w_exp, data)

    orig_min_x, orig_max_x, orig_min_y, orig_max_y, orig_plot_info = range_test_true_and_learned_vals(data)

    overall_min_x = min((w_v_min_x, sigma_min_x, both_min_x, orig_min_x))
    overall_max_x = max((w_v_max_x, sigma_max_x, both_max_x, orig_max_x))
    overall_min_y = min((w_v_min_y, sigma_min_y, both_min_y, orig_min_y))
    overall_max_y = max((w_v_max_y, sigma_max_y, both_max_y, orig_max_y))

    for p in w_v_plot_info:
        p.plot(overall_min_x, overall_max_x, overall_min_y, overall_max_y)
    for p in sigma_plot_info:
        p.plot(overall_min_x, overall_max_x, overall_min_y, overall_max_y)
    for p in both_plot_info:
        p.plot(overall_min_x, overall_max_x, overall_min_y, overall_max_y)
    for p in orig_plot_info:
        p.plot(overall_min_x, overall_max_x, overall_min_y, overall_max_y)

if __name__ == "__main__":
    main()