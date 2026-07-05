from eval_regression import *
from regression_trained_richardson import *

class EvalRegressionUsingTrainedRichardson(EvalRegression):
    def __init__(self):
        super().__init__()
        self.reg = RegressionTrainedRichardson()

    def outputPredictions(self, data_x, data_y, pred_y, target_y, csvFilename):
        with open(csvFilename, 'w', newline='') as csvfile:
                                                                                                                                                                                                                                                                                                                                                                                

            fieldnames = ["dummy", "index", "donki_date", "cdaw_date", "donki_speed", "donki_ha", "longitude", "latitude", "Accel", "2nd_order_speed_final", "2nd_order_speed_20R", "Central_PA", "MPA", "sunspots", "halo", "target", "100MeV_peak_intensity", "100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln", "threshold_time", "peak_time", "expected_richardson", "expected_richardson_ln", "Type_2_Area", "trained_richardson_ln", "diffusive_shock", "V log V", "CMEs_past_month", "CMEs_past_9_hours", "CMEs_over_1000_past_9_hrs", "Max_speed_past_day", "solar_wind_speed", "connection_angle_degrees", "connection_angle_degrees_phi_2_solar_wind_sq_div"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()        
        
            for i in range(0, len(data_x)):
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
                    "100MeV_peak_intensity_ln" : data_y[i][0],
                    "predicted_100MeV_peak_intensity_ln" : pred_y[i][0],
                    "threshold_time": data_x[i]["threshold_time"],
                    "peak_time": data_x[i]["peak_time"],
                    "expected_richardson": data_x[i]["expected_richardson"],
                    "expected_richardson_ln": data_x[i]["expected_richardson_ln"],
                    "Type_2_Area" : data_x[i]["Type_2_Area"],
                    "trained_richardson_ln" : data_x[i]["trained_richardson_ln"],
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

    def regNN(self):
        resFolder = "../res/trained"
        outFolder = "../out/using_trained_richardson/regNN_retrained"
        evalFolder = "../eval/using_trained_richardson/regNN_retrained"

        all_trainingData_x, all_trainingData_y  = self.reg.getFirstStageInput("{}/firstStageAllTraining.csv".format(resFolder))

        adam_model_001_1_retrained_checkpoint_path = "{}/adam_model_001_1_retrained.ckpt".format(outFolder)

        retrained_adamLearningRate = 0.001
        retrained_adamEpsilon = 1.0
        retrained_alpha = 0.3

        retrained_logFilename = "{}/adam_model_001_1_retrained.csv".format(outFolder)

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path

        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.regNN(retrained_adamLearningRate, all_trainingData_x, all_trainingData_y, retrained_logFilename, seed=1234, adamEpsilon=retrained_adamEpsilon, alpha=retrained_alpha, weights_filename=retrained_weights_filename)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInput("{}/firstStageTest.csv".format(resFolder))
        testData_target = self.reg.getTargets("{}/firstStageTest.csv".format(resFolder))

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)
        regNN_corr_sep, regNN_corr_sep_elevated, regNN_meanAbsoluteErrorSEP, regNN_meanAbsoluteErrorAll, regNN_FP, regNN_FN, regNN_TP, regNN_TN, regNN_F1, regNN_HSS, regNN_TSS = self.evaluate(testData_y, predictedData_y, testData_target)
        minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}/Predictions_Reg_NN_F1_{}.png".format(evalFolder, regNN_F1), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
        indexes = get_indexes("{}/firstStageTest.csv".format(resFolder))
        self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}/Predictions_Reg_NN_F1_{}.csv".format(evalFolder, regNN_F1))

        return adam_model_001_1_retrained

    def regNN_iterate(self):
        resFolder = "../res/trained"
        outFolder = "../out/using_trained_richardson/regNN_retrained"
        evalFolder = "../eval/using_trained_richardson/regNN_retrained/iterate"

        seenF1Scores = self.results.getSeenF1Scores("reg", 0)
        seenF1Scores.clear()

        retrained_adamLearningRate = 0.001
        retrained_adamEpsilon = 1.0
        retrained_alpha = 0.3

        numIterations = 5
        corrSEPMean = 0.0
        corrSEPElevatedMean = 0.0
        maeSEPMean = 0.0
        maeAllMean = 0.0
        FPMean = 0.0
        FNMean = 0.0
        TPMean = 0.0
        TNMean = 0.0
        F1Mean = 0.0
        HSSMean = 0.0
        TSSMean = 0.0

        for j in range(0, numIterations):
            retrained_weights_filename = "{}/it_{}.ckpt".format(outFolder, j)

            adam_model_001_1_retrained, _ = self.reg.regNN(retrained_adamLearningRate, None, None, None, seed=1234+j, adamEpsilon=retrained_adamEpsilon, alpha=retrained_alpha, weights_filename=retrained_weights_filename)

            # Evaluate
            testData_x, testData_y  = self.reg.getFirstStageInput("{}/firstStageTest.csv".format(resFolder))
            testData_target = self.reg.getTargets("{}/firstStageTest.csv".format(resFolder))

            predictedData_y = adam_model_001_1_retrained.predict(testData_x)
            regNN_corr_sep, regNN_corr_sep_elevated, regNN_meanAbsoluteErrorSEP, regNN_meanAbsoluteErrorAll, regNN_FP, regNN_FN, regNN_TP, regNN_TN, regNN_F1, regNN_HSS, regNN_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            evalFilename = getFilename("{}/Predictions_Reg_NN_F1_".format(evalFolder), regNN_F1, seenF1Scores, trailer="")
            minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
            indexes = get_indexes("{}/firstStageTest.csv".format(resFolder))
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename))

            corrSEPMean = corrSEPMean + regNN_corr_sep
            corrSEPElevatedMean = corrSEPElevatedMean + regNN_corr_sep_elevated
            maeSEPMean = maeSEPMean + regNN_meanAbsoluteErrorSEP
            maeAllMean = maeAllMean + regNN_meanAbsoluteErrorAll
            FPMean = FPMean + regNN_FP
            FNMean = FNMean + regNN_FN
            TPMean = TPMean + regNN_TP
            TNMean = TNMean + regNN_TN
            F1Mean = F1Mean + regNN_F1
            HSSMean= HSSMean + regNN_HSS
            TSSMean = TSSMean + regNN_TSS

            self.results.add_all_metric("reg", 0, {
                "corrSEP": regNN_corr_sep,
                "corrSEPElevated": regNN_corr_sep_elevated,
                "maeSEP": regNN_meanAbsoluteErrorSEP,
                "maeAll": regNN_meanAbsoluteErrorAll,
                "FP": regNN_FP,
                "FN": regNN_FN,
                "TP": regNN_TP,
                "TN": regNN_TN,
                "F1": regNN_F1,
                "HSS": regNN_HSS,
                "TSS": regNN_TSS
                })

        corrSEPMean = corrSEPMean / numIterations
        corrSEPElevatedMean = corrSEPElevatedMean / numIterations
        maeSEPMean = maeSEPMean / numIterations
        maeAllMean = maeAllMean / numIterations
        FPMean = FPMean / numIterations
        FNMean = FNMean / numIterations
        TPMean = TPMean / numIterations
        TNMean = TNMean / numIterations
        F1Mean = F1Mean / numIterations
        HSSMean = HSSMean / numIterations
        TSSMean = TSSMean / numIterations

        self.results.updateMetrics("reg", 0, {
            "corrSEPMean": corrSEPMean,
            "corrSEPElevatedMean": corrSEPElevatedMean,
            "maeSEPMean": maeSEPMean,
            "maeAllMean": maeAllMean,
            "FPMean": FPMean,
            "FNMean": FNMean,
            "TPMean": TPMean,
            "TNMean": TNMean,
            "F1Mean": F1Mean,
            "HSSMean": HSSMean,
            "TSSMean": TSSMean
            })

    def regNN_oversampled(self, oversampleAmount, testDataFilename, adamLearningRate, adam_epsilon, alpha):
        resFolder = "../res/trained"
        folder_trailer = int(oversampleAmount / 10)
        outFolder = "../out/using_trained_richardson/retrained_regNN_oversampled_0_{}".format(folder_trailer)
        evalFolder = "../eval/using_trained_richardson/retrained_regNN_oversampled_0_{}".format(folder_trailer)

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1Scores = self.results.getSeenF1Scores("reg", oversampleAmount)
        seenF1Scores.clear()
        min_x = None
        max_x = None
        min_y = None
        max_y = None

        numIterations = 5
        corrSEPMean = 0.0
        corrSEPElevatedMean = 0.0
        maeSEPMean = 0.0
        maeAllMean = 0.0
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

            NNModel, _ = self.reg.regNN(adamLearningRate, None, None, None, seed=1234+j, adamEpsilon=adam_epsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = NNModel.predict(testData_x)
            regNN_corr_sep, regNN_corr_sep_elevated, regNN_meanAbsoluteErrorSEP, regNN_meanAbsoluteErrorAll, reg_FP, reg_FN, reg_TP, reg_TN, reg_F1, reg_HSS, reg_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            corrSEPMean = corrSEPMean + regNN_corr_sep
            corrSEPElevatedMean = corrSEPElevatedMean + regNN_corr_sep_elevated
            maeSEPMean = maeSEPMean + regNN_meanAbsoluteErrorSEP
            maeAllMean = maeAllMean + regNN_meanAbsoluteErrorAll
            FPMean = FPMean + reg_FP
            FNMean = FNMean + reg_FN
            TPMean = TPMean + reg_TP
            TNMean = TNMean + reg_TN
            F1Mean = F1Mean + reg_F1
            HSSMean= HSSMean + reg_HSS
            TSSMean = TSSMean + reg_TSS

            self.results.add_all_metric("reg", oversampleAmount, {
                "corrSEP": regNN_corr_sep,
                "corrSEPElevated": regNN_corr_sep_elevated,
                "maeSEP": regNN_meanAbsoluteErrorSEP,
                "maeAll": regNN_meanAbsoluteErrorAll,
                "FP": reg_FP,
                "FN": reg_FN,
                "TP": reg_TP,
                "TN": reg_TN,
                "F1": reg_F1,
                "HSS": reg_HSS,
                "TSS": reg_TSS
                })

            evalFilename = getFilename("{}/Predictions_F1_".format(evalFolder), reg_F1, seenF1Scores, trailer="")
            minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename))

            if min_x is None:
                min_x = minX
            else:
                min_x = min(min_x, minX)
            if max_x is None:
                max_x = maxX
            else:
                max_x = max(max_x, maxX)
            if min_y is None:
                min_y = minY
            else:
                min_y = min(min_y, minY)
            if max_y is None:
                max_y = maxY
            else:
                max_y = max(max_y, maxY)


        self.results.updateMinX("reg", oversampleAmount, min_x)
        self.results.updateMaxX("reg", oversampleAmount, max_x)
        self.results.updateMinY("reg", oversampleAmount, min_y)
        self.results.updateMaxY("reg", oversampleAmount, max_y)

        corrSEPMean = corrSEPMean / numIterations
        corrSEPElevatedMean = corrSEPElevatedMean / numIterations
        maeSEPMean = maeSEPMean / numIterations
        maeAllMean = maeAllMean / numIterations
        FPMean = FPMean / numIterations
        FNMean = FNMean / numIterations
        TPMean = TPMean / numIterations
        TNMean = TNMean / numIterations
        F1Mean = F1Mean / numIterations
        HSSMean = HSSMean / numIterations
        TSSMean = TSSMean / numIterations

        self.results.updateMetrics("reg", oversampleAmount, {
            "corrSEPMean": corrSEPMean,
            "corrSEPElevatedMean": corrSEPElevatedMean,
            "maeSEPMean": maeSEPMean,
            "maeAllMean": maeAllMean,
            "FPMean": FPMean,
            "FNMean": FNMean,
            "TPMean": TPMean,
            "TNMean": TNMean,
            "F1Mean": F1Mean,
            "HSSMean": HSSMean,
            "TSSMean": TSSMean
            })

        seenF1Scores.clear()

        for j in range(0, 5):
            weights_filename = "{}/it_{}.ckpt".format(outFolder, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))
            NNModel, _ = self.reg.regNN(adamLearningRate, None, None, None, seed=1234+j, adamEpsilon=adam_epsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = NNModel.predict(testData_x)
            regNN_corr_sep, regNN_corr_sep_elevated, regNN_meanAbsoluteErrorSEP, regNN_meanAbsoluteErrorAll, reg_FP, reg_FN, reg_TP, reg_TN, reg_F1, reg_HSS, reg_TSS = self.evaluate(testData_y, predictedData_y, testData_target)
            evalFilename = getFilename("{}/Scaled_Predictions_F1_".format(evalFolder), reg_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), min_x, max_x, min_y, max_y, plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")

    def rRT(self, oversampleAmount, testDataFilename, adamLearningRate, adam_epsilon, alpha):
        resFolder = "../res/trained"
        folder_trailer = int(oversampleAmount / 10)
        outFolder = "../out/using_trained_richardson/retrained_rRT_0_{}".format(folder_trailer)
        evalFolder = "../eval/using_trained_richardson/retrained_rRT_0_{}".format(folder_trailer)

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1Scores = self.results.getSeenF1Scores("rRT", oversampleAmount)
        seenF1Scores.clear()
        min_x = None
        max_x = None
        min_y = None
        max_y = None

        numIterations = 5
        corrSEPMean = 0.0
        corrSEPElevatedMean = 0.0
        maeSEPMean = 0.0
        maeAllMean = 0.0
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
            retrained_weights_filename = "../out/using_trained_richardson/regNN_retrained/adam_model_001_1_retrained.ckpt"
            adam_model_001_1_retrained, _ = self.reg.regNN(retrained_adamLearningRate, None, None, None, seed=1234, adamEpsilon=retrained_adamEpsilon, alpha=retrained_alpha, weights_filename=retrained_weights_filename)

            feature_extractor = get_feature_extractor(adam_model_001_1_retrained)
            NNModel, _ = self.reg.rRT(adamLearningRate, feature_extractor, None, None, None, seed=1234+j, adamEpsilon=adam_epsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = NNModel.predict(testData_x)
            rRT_corr_sep, rRT_corr_sep_elevated, rRT_meanAbsoluteErrorSEP, rRT_meanAbsoluteErrorAll, rRT_FP, rRT_FN, rRT_TP, rRT_TN, rRT_F1, rRT_HSS, rRT_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            corrSEPMean = corrSEPMean + rRT_corr_sep
            corrSEPElevatedMean = corrSEPElevatedMean + rRT_corr_sep_elevated
            maeSEPMean = maeSEPMean + rRT_meanAbsoluteErrorSEP
            maeAllMean = maeAllMean + rRT_meanAbsoluteErrorAll
            FPMean = FPMean + rRT_FP
            FNMean = FNMean + rRT_FN
            TPMean = TPMean + rRT_TP
            TNMean = TNMean + rRT_TN
            F1Mean = F1Mean + rRT_F1
            HSSMean= HSSMean + rRT_HSS
            TSSMean = TSSMean + rRT_TSS

            self.results.add_all_metric("rRT", oversampleAmount, {
                "corrSEP": rRT_corr_sep,
                "corrSEPElevated": rRT_corr_sep_elevated,
                "maeSEP": rRT_meanAbsoluteErrorSEP,
                "maeAll": rRT_meanAbsoluteErrorAll,
                "FP": rRT_FP,
                "FN": rRT_FN,
                "TP": rRT_TP,
                "TN": rRT_TN,
                "F1": rRT_F1,
                "HSS": rRT_HSS,
                "TSS": rRT_TSS
                })

            evalFilename = getFilename("{}/Predictions_F1_".format(evalFolder), rRT_F1, seenF1Scores, trailer="")
            minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename))

            if min_x is None:
                min_x = minX
            else:
                min_x = min(min_x, minX)
            if max_x is None:
                max_x = maxX
            else:
                max_x = max(max_x, maxX)
            if min_y is None:
                min_y = minY
            else:
                min_y = min(min_y, minY)
            if max_y is None:
                max_y = maxY
            else:
                max_y = max(max_y, maxY)


        self.results.updateMinX("rRT", oversampleAmount, min_x)
        self.results.updateMaxX("rRT", oversampleAmount, max_x)
        self.results.updateMinY("rRT", oversampleAmount, min_y)
        self.results.updateMaxY("rRT", oversampleAmount, max_y)

        corrSEPMean = corrSEPMean / numIterations
        corrSEPElevatedMean = corrSEPElevatedMean / numIterations
        maeSEPMean = maeSEPMean / numIterations
        maeAllMean = maeAllMean / numIterations
        FPMean = FPMean / numIterations
        FNMean = FNMean / numIterations
        TPMean = TPMean / numIterations
        TNMean = TNMean / numIterations
        F1Mean = F1Mean / numIterations
        HSSMean = HSSMean / numIterations
        TSSMean = TSSMean / numIterations

        self.results.updateMetrics("rRT", oversampleAmount, {
            "corrSEPMean": corrSEPMean,
            "corrSEPElevatedMean": corrSEPElevatedMean,
            "maeSEPMean": maeSEPMean,
            "maeAllMean": maeAllMean,
            "FPMean": FPMean,
            "FNMean": FNMean,
            "TPMean": TPMean,
            "TNMean": TNMean,
            "F1Mean": F1Mean,
            "HSSMean": HSSMean,
            "TSSMean": TSSMean
            })

        seenF1Scores.clear()

        for j in range(0, 5):
            weights_filename = "{}/it_{}.ckpt".format(outFolder, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))
            
            retrained_adamLearningRate = 0.001
            retrained_adamEpsilon = 1.0
            retrained_alpha = 0.3
            retrained_weights_filename = "../out/using_trained_richardson/regNN_retrained/adam_model_001_1_retrained.ckpt"
            adam_model_001_1_retrained, _ = self.reg.regNN(retrained_adamLearningRate, None, None, None, seed=1234, adamEpsilon=retrained_adamEpsilon, alpha=retrained_alpha, weights_filename=retrained_weights_filename)

            feature_extractor = get_feature_extractor(adam_model_001_1_retrained)
            NNModel, _ = self.reg.rRT(adamLearningRate, feature_extractor, None, None, None, seed=1234+j, adamEpsilon=adam_epsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = NNModel.predict(testData_x)
            rRT_corr_sep, rRT_corr_sep_elevated, rRT_meanAbsoluteErrorSEP, rRT_meanAbsoluteErrorAll, rRT_FP, rRT_FN, rRT_TP, rRT_TN, rRT_F1, rRT_HSS, rRT_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            evalFilename = getFilename("{}/Scaled_Predictions_F1_".format(evalFolder), rRT_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), min_x, max_x, min_y, max_y, plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")

    def autoencoder(self, oversampleAmount, testDataFilename, adamLearningRate, adam_epsilon, alpha):
        resFolder = "../res/trained"
        folder_trailer = int(oversampleAmount / 10)
        outFolder = "../out/using_trained_richardson/retrained_autoencoder_ss_0_{}".format(folder_trailer)
        evalFolder = "../eval/using_trained_richardson/retrained_autoencoder_ss_0_{}".format(folder_trailer)

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1Scores = self.results.getSeenF1Scores("aut", oversampleAmount)
        seenF1Scores.clear()
        min_x = None
        max_x = None
        min_y = None
        max_y = None

        numIterations = 5
        corrSEPMean = 0.0
        corrSEPElevatedMean = 0.0
        maeSEPMean = 0.0
        maeAllMean = 0.0
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

            autoencoder_validation_weights_path = "../out/using_trained_richardson/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
            adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)
            adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)

            autoencoder_second_stage_model = self.reg.autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, None, None, None, None, alpha=alpha, weights_filename=weights_filename, train=False, seed=1234+j)
            predictedData_y = autoencoder_second_stage_model.predict(testData_x)
            aut_corr_sep, aut_corr_sep_elevated, aut_meanAbsoluteErrorSEP, aut_meanAbsoluteErrorAll, aut_FP, aut_FN, aut_TP, aut_TN, aut_F1, aut_HSS, aut_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            corrSEPMean = corrSEPMean + aut_corr_sep
            corrSEPElevatedMean = corrSEPElevatedMean + aut_corr_sep_elevated
            maeSEPMean = maeSEPMean + aut_meanAbsoluteErrorSEP
            maeAllMean = maeAllMean + aut_meanAbsoluteErrorAll
            FPMean = FPMean + aut_FP
            FNMean = FNMean + aut_FN
            TPMean = TPMean + aut_TP
            TNMean = TNMean + aut_TN
            F1Mean = F1Mean + aut_F1
            HSSMean= HSSMean + aut_HSS
            TSSMean = TSSMean + aut_TSS

            self.results.add_all_metric("aut", oversampleAmount, {
                "corrSEP": aut_corr_sep,
                "corrSEPElevated": aut_corr_sep_elevated,
                "maeSEP": aut_meanAbsoluteErrorSEP,
                "maeAll": aut_meanAbsoluteErrorAll,
                "FP": aut_FP,
                "FN": aut_FN,
                "TP": aut_TP,
                "TN": aut_TN,
                "F1": aut_F1,
                "HSS": aut_HSS,
                "TSS": aut_TSS
                })

            evalFilename = getFilename("{}/Predictions_F1_".format(evalFolder), aut_F1, seenF1Scores, trailer="")
            minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename))

            if min_x is None:
                min_x = minX
            else:
                min_x = min(min_x, minX)
            if max_x is None:
                max_x = maxX
            else:
                max_x = max(max_x, maxX)
            if min_y is None:
                min_y = minY
            else:
                min_y = min(min_y, minY)
            if max_y is None:
                max_y = maxY
            else:
                max_y = max(max_y, maxY)


        self.results.updateMinX("aut", oversampleAmount, min_x)
        self.results.updateMaxX("aut", oversampleAmount, max_x)
        self.results.updateMinY("aut", oversampleAmount, min_y)
        self.results.updateMaxY("aut", oversampleAmount, max_y)

        corrSEPMean = corrSEPMean / numIterations
        corrSEPElevatedMean = corrSEPElevatedMean / numIterations
        maeSEPMean = maeSEPMean / numIterations
        maeAllMean = maeAllMean / numIterations
        FPMean = FPMean / numIterations
        FNMean = FNMean / numIterations
        TPMean = TPMean / numIterations
        TNMean = TNMean / numIterations
        F1Mean = F1Mean / numIterations
        HSSMean = HSSMean / numIterations
        TSSMean = TSSMean / numIterations

        self.results.updateMetrics("aut", oversampleAmount, {
            "corrSEPMean": corrSEPMean,
            "corrSEPElevatedMean": corrSEPElevatedMean,
            "maeSEPMean": maeSEPMean,
            "maeAllMean": maeAllMean,
            "FPMean": FPMean,
            "FNMean": FNMean,
            "TPMean": TPMean,
            "TNMean": TNMean,
            "F1Mean": F1Mean,
            "HSSMean": HSSMean,
            "TSSMean": TSSMean
            })

        seenF1Scores.clear()

        for j in range(0, 5):
            weights_filename = "{}/it_{}.ckpt".format(outFolder, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))

            autoencoder_validation_weights_path = "../out/using_trained_richardson/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
            adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)
            adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)

            autoencoder_second_stage_model = self.reg.autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, None, None, None, None, alpha=alpha, weights_filename=weights_filename, train=False, seed=1234+j)
            predictedData_y = autoencoder_second_stage_model.predict(testData_x)
            aut_corr_sep, aut_corr_sep_elevated, aut_meanAbsoluteErrorSEP, aut_meanAbsoluteErrorAll, aut_FP, aut_FN, aut_TP, aut_TN, aut_F1, aut_HSS, aut_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            evalFilename = getFilename("{}/Scaled_Predictions_F1_".format(evalFolder), aut_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), min_x, max_x, min_y, max_y, plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")

if __name__ == "__main__":
    evaluator = EvalRegressionUsingTrainedRichardson()
    evaluator.regNN()
    evaluator.regNN_iterate()

    for i in range(1, 10):
        oversample_amount = 10 * i

        adamLearningRate = 0.001
        adamEpsilon = 1.0
        alpha = 0.3
        evaluator.regNN_oversampled(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(i), adamLearningRate, adamEpsilon, alpha)

        adamLearningrate = 0.001
        adamepsilon = 1.0
        alpha = 0.3
        evaluator.rRT(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(i), adamLearningRate, adamEpsilon, alpha)

        adamLearningRate = 0.001
        adamEpsilon = 1.0
        alpha = 0.3
        evaluator.autoencoder(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(i), adamLearningRate, adamEpsilon, alpha)

    evaluator.results.print_metrics()
    evaluator.results.print_F1()
