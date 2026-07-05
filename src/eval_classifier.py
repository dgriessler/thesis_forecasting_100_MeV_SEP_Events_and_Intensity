from base_eval import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from classifier import *
from results_classifier import *
from scipy.stats import pearsonr

class EvalClassifier(object):
    def __init__(self):
        self.cla = Classifier()
        self.orig_data = get_orig_data()
        self.results = ResultsClassifier()

    def evaluate(self, y_true, predictions, y_target, threshold):
        minX = min(y_true) - 0.1
        maxX = max(y_true) + 0.1
        minY = min(predictions) - 0.1
        maxY = max([max(predictions) + 0.1, 1.1])

        scatterColors = []
        for i in range(0, len(y_target)):
            target = y_target[i]
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

        corr, _ = pearsonr(np.reshape(y_true, len(y_true)), np.reshape(predictions, len(predictions)))
    
        # Is CME related to SEP Event | Actual No | Actual Yes
        # Predicted No                | TN        | FN
        # Predicted Yes               | FP        | TP
        TN = 0
        FN = 0
        FP = 0
        TP = 0
        for i in range(0, len(y_true)):
            actualY = y_true[i]
            predictedY = predictions[i]
            if actualY == 1:
                # Actual Yes
                if predictedY > threshold:
                    # Predicted Yes
                    TP = TP + 1
                else:
                    # Predictred No
                    FN = FN + 1
            else:
                # Actual No
                if predictedY > threshold:
                    # Predicted Yes
                    FP = FP + 1
                else:
                    # Predicted No
                    TN = TN + 1

        precisionDenominator = TP + FP
        if math.isclose(precisionDenominator, 0.0):
            precision = 0
        else:
            precision = TP / precisionDenominator

        # recall = TPR = TP / (TP + FN)
        recallDenominator = TP + FN
        if math.isclose(recallDenominator, 0.0):
            recall = 0
        else:
            recall = TP / recallDenominator
        TPR = recall
    
        F1Denominator = precision + recall
        if math.isclose(F1Denominator, 0.0):
            F1 = 0
        else:
            F1 = 2 * (precision * recall) / F1Denominator
    
        FPRDenominator = FP + TN
        if math.isclose(F1Denominator, 0.0):
            FPR = 0
        else:
            FPR = FP / FPRDenominator

        TSS = TPR - FPR
    
        HSSDenominator = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
        if math.isclose(HSSDenominator, 0.0):
            HSS = 0
        else:
            HSS = 2 * (TP * TN - FP * FN) / HSSDenominator
    
        return (corr, FP, FN, TP, TN, F1, HSS, TSS)

    def plotEvalGraph(self, y_true, predictions, y_target, filename, plotTitle="", xLabel="Classification", yLabel="Score", threshold=0.5):
        minX = min(y_true) - 0.1
        maxX = max(y_true) + 0.1
        minY = min(predictions) - 0.1
        maxY = max(predictions) + 0.1

        self.plotEvalGraphScaled(y_true, predictions, y_target, filename, minX, maxX, minY, maxY, plotTitle, xLabel, yLabel, threshold)

        return (minX, maxX, minY, maxY)

    def plotEvalGraphScaled(self, y_true, predictions, y_target, filename, minX, maxX, minY, maxY, plotTitle="", xLabel="Classification", yLabel="Score", threshold=0.5):
        # Plot y-axis as predicted value and x-axis as original value
        fig, ax = plt.subplots()

        ax.set_title(plotTitle)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_xlim(minX, maxX)
        ax.set_ylim(minY, maxY)

        scatters = []
        scatter_labels = ["Background", "Elevated", "SEP"]

        for particular_target in [0, 2, 1]:
            particular_x = []
            particular_y = []
            scatterColors = []
            actual_particular_x = []

            for i in range(0, len(y_target)):
                target = y_target[i]
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

                if target == particular_target:
                    particular_x.append(y_true[i])
                    particular_y.append(predictions[i])
                    if y_true[i] == 1:
                        actual_particular_x.append("SEP")
                    else:
                        actual_particular_x.append("Non-SEP")
                    scatterColors.append(color)

            scatter = ax.scatter(actual_particular_x, particular_y, c=scatterColors)
            scatters.append(scatter)

        thresholdHorizontalLine = mlines.Line2D([minX, maxX], [threshold, threshold], color='black', ls="--")

        ax.add_line(thresholdHorizontalLine)
        ax.legend(scatters, scatter_labels)

        plt.savefig(filename)
        plt.close(fig)

    def outputPredictions(self, data_x, data_y, pred_y, target_y, csvFilename, pred_threshold):
        with open(csvFilename, 'w', newline='') as csvfile:

            fieldnames = ["dummy", "index", "donki_date", "cdaw_date", "donki_speed", "donki_ha", "longitude", "latitude", "Accel", "2nd_order_speed_final", "2nd_order_speed_20R", "Central_PA", "MPA", "sunspots", "halo", "target", "100MeV_peak_intensity", "100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln", "predicted_thresholded", "threshold_time", "peak_time", "expected_richardson", "expected_richardson_ln", "Type_2_Area", "richardson_formula_degrees_phi_2_solar_wind", "diffusive_shock", "V log V", "CMEs_past_month", "CMEs_past_9_hours", "CMEs_over_1000_past_9_hrs", "Max_speed_past_day", "solar_wind_speed", "connection_angle_degrees", "connection_angle_degrees_phi_2_solar_wind_sq_div"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(0, len(data_x)):
                thresholded_value = 1
                if pred_y[i][0] < pred_threshold:
                    thresholded_value = 0
                row = {
                    "dummy" : 0,
                    "index" : data_x[i]["index"],
                    "donki_date": data_x[i]["donki_date"],
                    "cdaw_date": data_x[i]["cdaw_date"],
                    "donki_speed" : data_x[i]["donki_speed"],
                    "donki_ha" : data_x[i]["donki_ha"],
                    "longitude" : data_x[i]["longitude"],
                    "latitude" : data_x[i]["latitude"],
                    "Accel" : data_x[i]["Accel"],
                    "2nd_order_speed_final" : data_x[i]["2nd_order_speed_final"],
                    "2nd_order_speed_20R" : data_x[i]["2nd_order_speed_20R"],
                    "Central_PA" : data_x[i]["Central_PA"],
                    "MPA" : data_x[i]["MPA"],
                    "sunspots" : data_x[i]["sunspots"],
                    "halo" : data_x[i]["halo"],
                    "target" : target_y[i],
                    "100MeV_peak_intensity" : data_x[i]["100MeV_peak_intensity"],
                    "100MeV_peak_intensity_ln" : data_x[i]["100MeV_peak_intensity_ln"],
                    "predicted_100MeV_peak_intensity_ln" : pred_y[i][0],
                    "predicted_thresholded" : thresholded_value,
                    "threshold_time": data_x[i]["threshold_time"],
                    "peak_time": data_x[i]["peak_time"],
                    "expected_richardson": data_x[i]["expected_richardson"],
                    "expected_richardson_ln": data_x[i]["expected_richardson_ln"],
                    "Type_2_Area" : data_x[i]["Type_2_Area"],
                    "richardson_formula_degrees_phi_2_solar_wind" : data_x[i]["richardson_formula_degrees_phi_2_solar_wind"],
                    "diffusive_shock" : data_x[i]["diffusive_shock"],
                    "V log V" : data_x[i]["V log V"],
                    "CMEs_past_month" : data_x[i]["CMEs_past_month"],
                    "CMEs_past_9_hours" : data_x[i]["CMEs_past_9_hours"],
                    "CMEs_over_1000_past_9_hrs" : data_x[i]["CMEs_over_1000_past_9_hrs"],
                    "Max_speed_past_day" : data_x[i]["Max_speed_past_day"],
                    "solar_wind_speed" : data_x[i]["solar_wind_speed"],
                    "connection_angle_degrees" : data_x[i]["connection_angle_degrees"],
                    "connection_angle_degrees_phi_2_solar_wind_sq_div": data_x[i]["connection_angle_degrees_phi_2_solar_wind_sq_div"]    
                }
                writer.writerow(row)
                
        return

    def regNN(self, pred_threshold=0.5):
        resFolder = "../res/gen"
        outFolder = "../out/classifier/regNN_retrained"
        evalFolder = "../eval/classifier/regNN_retrained"

        all_trainingData_x, all_trainingData_y  = self.cla.getFirstStageInput("{}/firstStageAllTraining.csv".format(resFolder))

        adam_model_001_1_retrained_checkpoint_path = "{}/adam_model_001_1_retrained.ckpt".format(outFolder)

        retrained_adamLearningRate = 0.001
        retrained_adamEpsilon = 1.0
        retrained_alpha = 0.3

        retrained_logFilename = "{}/adam_model_001_1_retrained.csv".format(outFolder)

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path

        adam_model_001_1_retrained, adam_history_001_1_retrained = self.cla.regNN(retrained_adamLearningRate, all_trainingData_x, all_trainingData_y, retrained_logFilename, 1234, adamEpsilon=retrained_adamEpsilon, alpha=retrained_alpha, weights_filename=retrained_weights_filename)

        # Evaluate
        testData_x, testData_y  = self.cla.getFirstStageInput("{}/firstStageTest.csv".format(resFolder))
        testData_target = self.cla.getTargets("{}/firstStageTest.csv".format(resFolder))

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)
        regNN_corr, regNN_FP, regNN_FN, regNN_TP, regNN_TN, regNN_F1, regNN_HSS, regNN_TSS = self.evaluate(testData_y, predictedData_y, testData_target, pred_threshold)
        self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}/threshold_{}_Predictions_Reg_NN_F1_{}.png".format(evalFolder, pred_threshold, regNN_F1), -0.1, 1.1, -0.1, 1.1, threshold=pred_threshold)

        indexes = get_indexes("{}/firstStageTest.csv".format(resFolder))
        self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}/threshold_{}_Predictions_Reg_NN_F1_{}.csv".format(evalFolder, pred_threshold, regNN_F1), pred_threshold=pred_threshold)

        return adam_model_001_1_retrained

    def regNN_iterate(self, pred_threshold=0.5):
        resFolder = "../res/gen"
        outFolder = "../out/classifier/regNN_retrained"
        evalFolder = "../eval/classifier/regNN_retrained/iterate"

        seenF1Scores = self.results.getSeenF1Scores("reg", 0)
        seenF1Scores.clear()

        retrained_adamLearningRate = 0.001
        retrained_adamEpsilon = 1.0
        retrained_alpha = 0.3

        numIterations = 5
        corrMean = 0.0
        FPMean = 0.0
        FNMean = 0.0
        TPMean = 0.0
        TNMean = 0.0
        F1Mean = 0.0
        HSSMean = 0.0
        TSSMean = 0.0

        for j in range(0, numIterations):
            retrained_weights_filename = "{}/it_{}.ckpt".format(outFolder, j)

            adam_model_001_1_retrained, _ = self.cla.regNN(retrained_adamLearningRate, None, None, None, 1234+j, adamEpsilon=retrained_adamEpsilon, alpha=retrained_alpha, weights_filename=retrained_weights_filename)

            # Evaluate
            testData_x, testData_y  = self.cla.getFirstStageInput("{}/firstStageTest.csv".format(resFolder))
            testData_target = self.cla.getTargets("{}/firstStageTest.csv".format(resFolder))

            predictedData_y = adam_model_001_1_retrained.predict(testData_x)
            regNN_corr, regNN_FP, regNN_FN, regNN_TP, regNN_TN, regNN_F1, regNN_HSS, regNN_TSS = self.evaluate(testData_y, predictedData_y, testData_target, pred_threshold)      

            evalFilename = getFilename("{}/threshold_{}_Predictions_Reg_NN_F1_".format(evalFolder, pred_threshold), regNN_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), -0.1, 1.1, -0.1, 1.1, threshold=pred_threshold)
            indexes = get_indexes("{}/firstStageTest.csv".format(resFolder))
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename), pred_threshold=pred_threshold)

            corrMean = corrMean + regNN_corr
            FPMean = FPMean + regNN_FP
            FNMean = FNMean + regNN_FN
            TPMean = TPMean + regNN_TP
            TNMean = TNMean + regNN_TN
            F1Mean = F1Mean + regNN_F1
            HSSMean= HSSMean + regNN_HSS
            TSSMean = TSSMean + regNN_TSS

            self.results.add_all_metric("reg", 0, pred_threshold, {
                "corr": regNN_corr,
                "FP": regNN_FP,
                "FN": regNN_FN,
                "TP": regNN_TP,
                "TN": regNN_TN,
                "F1": regNN_F1,
                "HSS": regNN_HSS,
                "TSS": regNN_TSS
                })

        corrMean = corrMean / numIterations
        FPMean = FPMean / numIterations
        FNMean = FNMean / numIterations
        TPMean = TPMean / numIterations
        TNMean = TNMean / numIterations
        F1Mean = F1Mean / numIterations
        HSSMean = HSSMean / numIterations
        TSSMean = TSSMean / numIterations

        self.results.updateMetrics("reg", 0, pred_threshold, {
            "corrMean": corrMean,
            "FPMean": FPMean,
            "FNMean": FNMean,
            "TPMean": TPMean,
            "TNMean": TNMean,
            "F1Mean": F1Mean,
            "HSSMean": HSSMean,
            "TSSMean": TSSMean
            })

    def regNN_oversampled(self, oversampleAmount, testDataFilename, adamLearningRate, adam_epsilon, alpha, pred_threshold=0.5):
        resFolder = "../res/gen"
        folder_trailer = int(oversampleAmount / 10)
        outFolder = "../out/classifier/retrained_regNN_oversampled_0_{}".format(folder_trailer)
        evalFolder = "../eval/classifier/retrained_regNN_oversampled_0_{}".format(folder_trailer)

        testData_x, testData_y  = self.cla.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.cla.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1Scores = self.results.getSeenF1Scores("reg", oversampleAmount)
        seenF1Scores.clear()

        numIterations = 5
        corrMean = 0.0
        FPMean = 0.0
        FNMean = 0.0
        TPMean = 0.0
        TNMean = 0.0
        F1Mean = 0.0
        HSSMean = 0.0
        TSSMean = 0.0

        for j in range(0, numIterations):
            weights_filename = "{}/it_{}.ckpt".format(outFolder, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))
            NNModel, _ = self.cla.regNN(adamLearningRate, None, None, None, 1234+j, adamEpsilon=adam_epsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = NNModel.predict(testData_x)
            reg_corr, reg_FP, reg_FN, reg_TP, reg_TN, reg_F1, reg_HSS, reg_TSS = self.evaluate(testData_y, predictedData_y, testData_target, pred_threshold)

            corrMean = corrMean + reg_corr
            FPMean = FPMean + reg_FP
            FNMean = FNMean + reg_FN
            TPMean = TPMean + reg_TP
            TNMean = TNMean + reg_TN
            F1Mean = F1Mean + reg_F1
            HSSMean= HSSMean + reg_HSS
            TSSMean = TSSMean + reg_TSS

            self.results.add_all_metric("reg", oversampleAmount, pred_threshold, {
                "corr": reg_corr,
                "FP": reg_FP,
                "FN": reg_FN,
                "TP": reg_TP,
                "TN": reg_TN,
                "F1": reg_F1,
                "HSS": reg_HSS,
                "TSS": reg_TSS
                })

            evalFilename = getFilename("{}/threshold_{}_Predictions_F1_".format(evalFolder, pred_threshold), reg_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), -0.1, 1.1, -0.1, 1.1, threshold=pred_threshold)
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename), pred_threshold=pred_threshold)

        corrMean = corrMean / numIterations
        FPMean = FPMean / numIterations
        FNMean = FNMean / numIterations
        TPMean = TPMean / numIterations
        TNMean = TNMean / numIterations
        F1Mean = F1Mean / numIterations
        HSSMean = HSSMean / numIterations
        TSSMean = TSSMean / numIterations
        print("reg,{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(oversampleAmount, corrMean, FPMean, FNMean, TPMean, TNMean, F1Mean, HSSMean, TSSMean))

        self.results.updateMetrics("reg", oversampleAmount, pred_threshold, {
            "corrMean": corrMean,
            "FPMean": FPMean,
            "FNMean": FNMean,
            "TPMean": TPMean,
            "TNMean": TNMean,
            "F1Mean": F1Mean,
            "HSSMean": HSSMean,
            "TSSMean": TSSMean
            })

        seenF1Scores.clear()

    def rRT(self, oversampleAmount, testDataFilename, adamLearningRate, adam_epsilon, alpha, pred_threshold=0.5):
        resFolder = "../res/gen"
        folder_trailer = int(oversampleAmount / 10)
        outFolder = "../out/classifier/retrained_rRT_0_{}".format(folder_trailer)
        evalFolder = "../eval/classifier/retrained_rRT_0_{}".format(folder_trailer)

        testData_x, testData_y  = self.cla.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.cla.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1Scores = self.results.getSeenF1Scores("rRT", oversampleAmount)
        seenF1Scores.clear()

        numIterations = 5
        corrMean = 0.0
        FPMean = 0.0
        FNMean = 0.0
        TPMean = 0.0
        TNMean = 0.0
        F1Mean = 0.0
        HSSMean = 0.0
        TSSMean = 0.0

        for j in range(0, numIterations):
            weights_filename = "{}/it_{}.ckpt".format(outFolder, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))

            retrained_adamLearningRate = 0.001
            retrained_adamEpsilon = 1.0
            retrained_alpha = 0.3
            retrained_weights_filename = "../out/classifier/regNN_retrained/adam_model_001_1_retrained.ckpt"
            adam_model_001_1_retrained, _ = self.cla.regNN(retrained_adamLearningRate, None, None, None, 1234, adamEpsilon=retrained_adamEpsilon, alpha=retrained_alpha, weights_filename=retrained_weights_filename)

            NNModel, _ = self.cla.rRT(adamLearningRate, adam_model_001_1_retrained, None, None, None, 1234+j, adamEpsilon=adam_epsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = NNModel.predict(testData_x)
            rRT_corr, rRT_FP, rRT_FN, rRT_TP, rRT_TN, rRT_F1, rRT_HSS, rRT_TSS = self.evaluate(testData_y, predictedData_y, testData_target, pred_threshold)

            corrMean = corrMean + rRT_corr
            FPMean = FPMean + rRT_FP
            FNMean = FNMean + rRT_FN
            TPMean = TPMean + rRT_TP
            TNMean = TNMean + rRT_TN
            F1Mean = F1Mean + rRT_F1
            HSSMean= HSSMean + rRT_HSS
            TSSMean = TSSMean + rRT_TSS

            self.results.add_all_metric("rRT", oversampleAmount, pred_threshold, {
                "corr": rRT_corr,
                "FP": rRT_FP,
                "FN": rRT_FN,
                "TP": rRT_TP,
                "TN": rRT_TN,
                "F1": rRT_F1,
                "HSS": rRT_HSS,
                "TSS": rRT_TSS
                })       

            evalFilename = getFilename("{}/threshold_{}_Predictions_F1_".format(evalFolder, pred_threshold), rRT_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), -0.1, 1.1, -0.1, 1.1, threshold=pred_threshold)
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename), pred_threshold=pred_threshold)

        corrMean = corrMean / numIterations
        FPMean = FPMean / numIterations
        FNMean = FNMean / numIterations
        TPMean = TPMean / numIterations
        TNMean = TNMean / numIterations
        F1Mean = F1Mean / numIterations
        HSSMean = HSSMean / numIterations
        TSSMean = TSSMean / numIterations
        print("rRT,{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(oversampleAmount, corrMean, FPMean, FNMean, TPMean, TNMean, F1Mean, HSSMean, TSSMean))

        self.results.updateMetrics("rRT", oversampleAmount, pred_threshold, {
            "corrMean": corrMean,
            "FPMean": FPMean,
            "FNMean": FNMean,
            "TPMean": TPMean,
            "TNMean": TNMean,
            "F1Mean": F1Mean,
            "HSSMean": HSSMean,
            "TSSMean": TSSMean
            })

        seenF1Scores.clear()

    def autoencoder(self, oversampleAmount, testDataFilename, adamLearningRate, adam_epsilon, pred_threshold=0.5):
        resFolder = "../res/gen"
        folder_trailer = int(oversampleAmount / 10)
        outFolder = "../out/classifier/retrained_autoencoder_ss_0_{}".format(folder_trailer)
        evalFolder = "../eval/classifier/retrained_autoencoder_ss_0_{}".format(folder_trailer)

        testData_x, testData_y  = self.cla.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.cla.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1Scores = self.results.getSeenF1Scores("aut", oversampleAmount)
        seenF1Scores.clear()

        numIterations = 5
        corrMean = 0.0
        FPMean = 0.0
        FNMean = 0.0
        TPMean = 0.0
        TNMean = 0.0
        F1Mean = 0.0
        HSSMean = 0.0
        TSSMean = 0.0

        print("\n\n\n\n BEGIN \n\n\n\n")

        for j in range(0, numIterations):
            weights_filename = "{}/it_{}.ckpt".format(outFolder, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))

            autoencoder_validation_weights_path = "../out/classifier/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
            adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)
            adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)

            autoencoder_second_stage_model = self.cla.autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, None, None, None, None, alpha, weights_filename=weights_filename, train=False, seed=1234+j)
            predictedData_y = autoencoder_second_stage_model.predict(testData_x)
            aut_corr, aut_FP, aut_FN, aut_TP, aut_TN, aut_F1, aut_HSS, aut_TSS = self.evaluate(testData_y, predictedData_y, testData_target, pred_threshold)

            corrMean = corrMean + aut_corr
            FPMean = FPMean + aut_FP
            FNMean = FNMean + aut_FN
            TPMean = TPMean + aut_TP
            TNMean = TNMean + aut_TN
            F1Mean = F1Mean + aut_F1
            HSSMean= HSSMean + aut_HSS
            TSSMean = TSSMean + aut_TSS

            self.results.add_all_metric("aut", oversampleAmount, pred_threshold, {
                "corr": aut_corr,
                "FP": aut_FP,
                "FN": aut_FN,
                "TP": aut_TP,
                "TN": aut_TN,
                "F1": aut_F1,
                "HSS": aut_HSS,
                "TSS": aut_TSS
                })

            evalFilename = getFilename("{}/threshold_{}_Predictions_F1_".format(evalFolder, pred_threshold), aut_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), -0.1, 1.1, -0.1, 1.1, threshold=pred_threshold)
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename), pred_threshold=pred_threshold)

        corrMean = corrMean / numIterations
        FPMean = FPMean / numIterations
        FNMean = FNMean / numIterations
        TPMean = TPMean / numIterations
        TNMean = TNMean / numIterations
        F1Mean = F1Mean / numIterations
        HSSMean = HSSMean / numIterations
        TSSMean = TSSMean / numIterations
        print("aut,{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(oversampleAmount, corrMean, FPMean, FNMean, TPMean, TNMean, F1Mean, HSSMean, TSSMean))

        self.results.updateMetrics("aut", oversampleAmount, pred_threshold, {
            "corrMean": corrMean,
            "FPMean": FPMean,
            "FNMean": FNMean,
            "TPMean": TPMean,
            "TNMean": TNMean,
            "F1Mean": F1Mean,
            "HSSMean": HSSMean,
            "TSSMean": TSSMean
            })

        seenF1Scores.clear()

if __name__ == "__main__":
    evaluator = EvalClassifier()
    evaluator.regNN()

    for over_amount_round in range(0, 9):
        pred_threshold = round(0.1 * (over_amount_round + 1), 2)
        evaluator.regNN_iterate(pred_threshold=pred_threshold)

    for over_amount in range(1, 10):
        oversample_amount = 10 * over_amount

        for over_amount_round in range(0, 9):
            pred_threshold = round(0.1 * (over_amount_round + 1), 2)

            adamLearningRate = 0.001
            adamEpsilon = 1.0
            alpha = 0.3
            evaluator.regNN_oversampled(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(over_amount), adamLearningRate, adamEpsilon, alpha=alpha, pred_threshold=pred_threshold)

            adamLearningRate = 0.001
            adamEpsilon = 1.0
            alpha = 0.3
            evaluator.rRT(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(over_amount), adamLearningRate, adamEpsilon, alpha=alpha, pred_threshold=pred_threshold)

            adamLearningRate = 0.001
            adamEpsilon = 1.0
            evaluator.autoencoder(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(over_amount), adamLearningRate, adamEpsilon, pred_threshold=pred_threshold)


    evaluator.results.print_metrics()
    evaluator.results.print_F1()
