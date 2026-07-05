from base_eval import *
import matplotlib.pyplot as plt

class DL:
    def __init__(self):
        self.reg_oversampling = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        self.reg_F1 = [0.000, 0.000, 0.100, 0.100, 0.100, 0.257, 0.400, 0.457, 0.667, 0.744, 0.014, 0.014, 0.014]

        self.rRT_oversampling = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        self.rRT_F1 = [0.200, 0.314, 0.457, 0.500, 0.500, 0.533, 0.800, 0.800, 0.727, 0.626, 0.627, 0.530, 0.492]

        self.rRT_AE_oversampling = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        self.rRT_AE_F1 = [0.257, 0.314, 0.500, 0.500, 0.500, 0.533, 0.800, 0.800, 0.691, 0.638, 0.627, 0.514, 0.420]

        self.x_min = -0.1
        self.x_max = 2.1

        self.title = "F1 vs DW α"
        self.x_axis_label = "DW α"
        self.plot_labels = ["DL+rRegNN", "DL+rRT", "DL+rRT+AE"]

        self.save_file = "../eval/dl_F1_vs_Oversampling_Rate.png"

class Classifier:
    def __init__(self):
        self.reg_oversampling = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.reg_F1 = [0.000, 0.309, 0.598, 0.587, 0.497, 0.463, 0.441, 0.300, 0.196, 0.131]

        self.rRT_oversampling = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.rRT_F1 = [0.463, 0.467, 0.571, 0.541, 0.498, 0.641, 0.571, 0.564, 0.564]

        self.rRT_AE_oversampling = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.rRT_AE_F1 = [0.496, 0.622, 0.718, 0.756, 0.756, 0.771, 0.800, 0.800, 0.800]

        self.x_min = -10
        self.x_max = 100

        self.title = "F1 vs Oversampling Rate"
        self.x_axis_label = "Oversampling Rate"
        self.plot_labels = ["cRegNN", "cRT", "cRT+AE"]

        self.save_file = "../eval/classifier_F1_vs_Oversampling_Rate.png"

class Regression:
    def __init__(self):
        self.reg_oversampling = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.reg_F1 = [0.000, 0.000, 0.000, 0.448, 0.445, 0.440, 0.357, 0.545, 0.513, 0.454]

        self.rRT_oversampling = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.rRT_F1 = [0.730, 0.626, 0.615, 0.615, 0.646, 0.667, 0.667, 0.615, 0.549]

        self.rRT_AE_oversampling = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.rRT_AE_F1 = [0.742, 0.626, 0.615, 0.615, 0.636, 0.656, 0.650, 0.561, 0.270]

        self.x_min = -10
        self.x_max = 100

        self.title = "F1 vs Oversampling Rate"
        self.x_axis_label = "Oversampling Rate"
        self.plot_labels = ["rRegNN", "rRT", "rRT+AE"]

        self.save_file = "../eval/regression_F1_vs_Oversampling_Rate.png"

class RC:
    def __init__(self):
        self.reg_oversampling = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.reg_F1 = [0.050, 0.102, 0.040, 0.203, 0.129, 0.217, 0.153, 0.146, 0.163, 0.179]

        self.rRT_oversampling = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.rRT_F1 = [0.222, 0.222, 0.222, 0.222, 0.265, 0.382, 0.365, 0.252, 0.174]

        self.rRT_AE_oversampling = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.rRT_AE_F1 = [0.480, 0.583, 0.593, 0.588, 0.644, 0.553, 0.571, 0.539, 0.243]

        self.x_min = -10
        self.x_max = 100

        self.title = "F1 vs Oversampling Rate"
        self.x_axis_label = "Oversampling Rate"
        self.plot_labels = ["RC+rRegNN", "RC+rRT", "RC+rRT+AE"]

        self.save_file = "../eval/rc_F1_vs_Oversampling_Rate.png"

class RE:
    def __init__(self):
        self.reg_oversampling = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.reg_F1 = [0.194, 0.257, 0.238, 0.193, 0.193, 0.249, 0.369, 0.231, 0.144, 0.283]

        self.rRT_oversampling = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.rRT_F1 = [0.213, 0.479, 0.521, 0.456, 0.415, 0.435, 0.425, 0.321, 0.306]

        self.rRT_AE_oversampling = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.rRT_AE_F1 = [0.409, 0.545, 0.536, 0.427, 0.404, 0.406, 0.277, 0.308, 0.277]

        self.x_min = -10
        self.x_max = 100

        self.title = "F1 vs Oversampling Rate"
        self.x_axis_label = "Oversampling Rate"
        self.plot_labels = ["RE+rRegNN", "RE+rRT", "RE+rRT+AE"]

        self.save_file = "../eval/re_F1_vs_Oversampling_Rate.png"


def main():
    #a = [Regression(), RC(), RE(), DL(), Classifier()]
    a = [DL()]

    for elem in a:
        fig, ax = plt.subplots()
        y_min = -0.1
        y_max = 1.1
        ax.set_title(elem.title)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(elem.x_axis_label)
        ax.set_ylabel("F1")
        ax.set_xlim(elem.x_min, elem.x_max)

        ax.plot(elem.reg_oversampling, elem.reg_F1, label=elem.plot_labels[0]),
        ax.plot(elem.rRT_oversampling, elem.rRT_F1, label=elem.plot_labels[1]),
        ax.plot(elem.rRT_AE_oversampling, elem.rRT_AE_F1, label=elem.plot_labels[2])

        ax.legend()
        plt.savefig(elem.save_file)
        plt.close(fig)

def richardson_compare():
    orig_data = get_orig_data()
    hundred_mev_ln_intensity = []
    current_richardson_feature = []
    richardson_ln_conn_only = []
    scatter_colors = []
    for key in orig_data.keys():
        if int(orig_data[key]["target"]) > 0:
            if int(orig_data[key]["target"]) == 1:
                scatter_colors.append("red")
            else:
                scatter_colors.append("green")
            hundred_mev_ln_intensity.append(float(orig_data[key]["100MeV_peak_intensity_ln"]))
            current_richardson_feature.append(float(orig_data[key]["richardson_formula_degrees_phi_2_solar_wind"]))
            conn_sq = float(orig_data[key]["richardson_formula_degrees_ln"])
            neg_conn_sq = -1 * conn_sq
            neg_conn_sq_div_sigma = neg_conn_sq / (2 * 43 * 43)
            richardson_ln_conn_only.append(neg_conn_sq_div_sigma)

    min_current_richardson_feature = min(current_richardson_feature)
    max_current_richardson_feature = max(current_richardson_feature)
    normalized_current_richardson_feature = []
    for elem in current_richardson_feature:
        normalized_current_richardson_feature.append((elem - min_current_richardson_feature) / (max_current_richardson_feature - min_current_richardson_feature))

    min_richardson_ln_conn_only = min(richardson_ln_conn_only)
    max_richardson_ln_conn_only = max(richardson_ln_conn_only)
    normalized_richardson_ln_conn_only = []
    for elem in richardson_ln_conn_only:
        normalized_richardson_ln_conn_only.append((elem - min_richardson_ln_conn_only) / (max_richardson_ln_conn_only - min_richardson_ln_conn_only))

    fig, ax = plt.subplots()
    x_min = -0.1
    x_max = 1.1
    ax.set_title("richardson_formula_conn_only_normalized")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Conn Term Richardson Formula Normalized")
    ax.set_ylabel("100 MeV Peak Intensity LN")
    ax.set_ylim(-3, 5)

    x = normalized_current_richardson_feature
    y = hundred_mev_ln_intensity

    ax.scatter(x, y, c=scatter_colors)

    #calculate equation for trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    print(z)

    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    rsquared = ssreg / sstot
    print(rsquared)

    #add trendline to plot
    ax.plot(x, p(x), c='black', linestyle="dotted")

    ax.text(0.0, 3, "y = " + "{:.3e}".format(z[0]) + "x + " + "{:.3e}".format(z[1]), fontsize=12)

    plt.savefig("../eval/richardson_formula_conn_only.png")
    plt.close(fig)



    fig, ax = plt.subplots()
    x_min = -0.1
    x_max = 1.1
    ax.set_title("richardson_formula_ln_conn_only_neg_div_sigma_normalized")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Neg Conn Div Sigma Richardson LN Normalized")
    ax.set_ylabel("100 MeV Peak Intensity LN")
    ax.set_ylim(-3, 5)

    x = normalized_richardson_ln_conn_only
    y = hundred_mev_ln_intensity

    ax.scatter(x, y, c=scatter_colors)

    #calculate equation for trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    print(z)

    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    rsquared = ssreg / sstot
    print(rsquared)

    #add trendline to plot
    ax.plot(x, p(x), c='black', linestyle="dotted")

    ax.text(0.0, 3, "y = " + "{:.3e}".format(z[0]) + "x + " + "{:.3e}".format(z[1]), fontsize=12)

    plt.savefig("../eval/richardson_formula_ln_conn_only.png")
    plt.close(fig)

def validation_illustration():
    data = readCSVFile("../out/classifier/regNN/adam_model_001_1.csv")
    epoch = []
    loss = []
    val_loss = []
    for elem in data:
        epoch.append(int(elem["epoch"]))
        loss.append(float(elem["loss"]))
        val_loss.append(float(elem["val_loss"]))

    fig, ax = plt.subplots()
    ax.set_title("Error versus Epoch cRegNN 0% oversampling")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_ylim(0.0, 0.3)

    ax.scatter(epoch, loss, label="Loss")
    ax.scatter(epoch, val_loss, label="Val Loss")
    ax.legend()

    plt.savefig("../eval/error_versus_weight_updates_cRegNN_0.png")
    plt.close(fig)

def our_pareto_distribution():
    data = readCSVFile("../denseLoss/features_with_dense_loss_and_p_y.csv")
    one_hundred_mev_sep = []
    p_y_sep = []

    one_hundred_mev_elevated = []
    p_y_elevated = []

    one_hundred_mev_background = []
    p_y_background = []

    for elem in data:
        if int(elem["target"]) == 1:
            one_hundred_mev_sep.append(float(elem["100MeV_peak_intensity"]))
            p_y_sep.append(float(elem["p_y"]))
        elif int(elem["target"]) == 2:
            one_hundred_mev_elevated.append(float(elem["100MeV_peak_intensity"]))
            p_y_elevated.append(float(elem["p_y"]))
        else:
            one_hundred_mev_background.append(float(elem["100MeV_peak_intensity"]))
            p_y_background.append(float(elem["p_y"]))

    fig, ax = plt.subplots()
    ax.set_title("P(y) vs 100 MeV Peak Intensity")
    ax.set_xlabel("100 MeV Peak Intensity")
    ax.set_ylabel("P(y)")

    ax.scatter(one_hundred_mev_sep, p_y_sep, c='red', label="SEP")
    ax.scatter(one_hundred_mev_elevated, p_y_elevated, c='green', label="Elevated")
    ax.scatter(one_hundred_mev_background, p_y_background, c='blue', label="Background")
    ax.legend()

    plt.savefig("../eval/pareto_estimate.png")
    plt.close(fig)

main()