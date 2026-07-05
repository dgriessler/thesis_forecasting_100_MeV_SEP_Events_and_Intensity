from base_eval import *
from regression import *

class EvalRichardson(object):
    def __init__(self):
        self.reg = Regression()
        self.orig_data = get_orig_data()

    def evaluate(self, y_true, predictions, y_target, include_elevated = True, exp_before_mae = False):
        y_true_SEP = []
        predictions_SEP = []

        for i in range(0, len(y_target)):
            target = y_target[i]
            if target == 0:
                # Background
                pass
            elif target == 1:
                # SEP
                if exp_before_mae:
                    y_true_SEP.append(math.exp(y_true[i]))
                    predictions_SEP.append(math.exp(predictions[i]))
                else:
                    y_true_SEP.append(y_true[i])
                    predictions_SEP.append(predictions[i])
            elif target == 2:
                # Elevated
                if include_elevated:
                    if exp_before_mae:
                        y_true_SEP.append(math.exp(y_true[i]))
                        predictions_SEP.append(math.exp(predictions[i]))
                    else:
                        y_true_SEP.append(y_true[i])
                        predictions_SEP.append(predictions[i])

        meanAbsoluteErrorSEP = mae(y_true_SEP, predictions_SEP)
        corr, _ = pearsonr(np.reshape(y_true, len(y_true)), np.reshape(predictions, len(predictions)))
    
        return (corr, meanAbsoluteErrorSEP)

    def plotEvalGraph(self, y_true, predictions, y_target, filename, plotTitle="Predicted Value vs Original Value", xLabel="Original Value", yLabel="Predicted Value"):
        minX = min(y_true) - 0.1
        maxX = max(y_true) + 0.1
        minY = min(predictions) - 0.1
        maxY = max(predictions) + 0.1

        self.plotEvalGraphScaled(y_true, predictions, y_target, filename, minX, maxX, minY, maxY, plotTitle, xLabel, yLabel)

        return (minX, maxX, minY, maxY)

    def plotEvalGraphScaled(self, y_true, predictions, y_target, filename, minX, maxX, minY, maxY, plotTitle="Predicted Value vs Original Value", xLabel="Original Value", yLabel="Predicted Value"):
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

        # Plot y-axis as predicted value and x-axis as original value

        fig, ax = plt.subplots()

        ax.scatter(y_true, predictions, c=scatterColors)
        ax.set_title(plotTitle)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_xlim(minX, maxX)
        ax.set_ylim(minY, maxY)

        dottedDiagLine = mlines.Line2D([minX, maxY], [minX, maxY], color='black', ls="--")
        ax.add_line(dottedDiagLine)

        plt.savefig(filename)
        plt.close(fig)

    def outputPredictions(self, data_x, data_y, pred_y, target_y, csvFilename):
        with open(csvFilename, 'w', newline='') as csvfile:
            fieldnames = ["dummy", "index", "donki_date", "cdaw_date", "donki_speed", "donki_ha", "longitude", "latitude", "Accel", "Type_2_Area", "2nd_order_speed_final", "2nd_order_speed_20R", "Central_PA", "MPA", "sunspots", "CMEs_past_month", "CMEs_past_9_hours", "CMEs_over_1000_past_9_hrs", "Max_speed_past_day", "richardson_formula_degrees_phi_2_solar_wind", "V log V", "halo", "diffusive_shock", "Double_CME_100_MeV", "target", "100MeV_peak_intensity", "100MeV_peak_intensity_ln", "predicted_100MeV_peak_intensity_ln", "connection_angle_degrees", "connection_angle_degrees_phi_2_solar_wind_sq_div", "expected_richardson", "expected_richardson_ln"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()        
        
            for i in range(0, len(data_x)):
                row = {
                    "dummy" : 0,
                    "index" : (i + 1),
                    "donki_date" : data_x[i]["donki_date"],
                    "cdaw_date" : data_x[i]["cdaw_date"],
                    "donki_speed" : data_x[i]["donki_speed"],
                    "donki_ha" : data_x[i]["donki_ha"],
                    "longitude" : data_x[i]["longitude"],
                    "latitude" : data_x[i]["latitude"],
                    "Accel" : data_x[i]["Accel"],
                    "Type_2_Area" : data_x[i]["Type_2_Area"],
                    "2nd_order_speed_final" : data_x[i]["2nd_order_speed_final"],
                    "2nd_order_speed_20R" : data_x[i]["2nd_order_speed_20R"],
                    "Central_PA" : data_x[i]["Central_PA"],
                    "MPA" : data_x[i]["MPA"],
                    "sunspots" : data_x[i]["sunspots"],
                    "CMEs_past_month" : data_x[i]["CMEs_past_month"],
                    "CMEs_past_9_hours" : data_x[i]["CMEs_past_9_hours"],
                    "CMEs_over_1000_past_9_hrs" : data_x[i]["CMEs_over_1000_past_9_hrs"],
                    "Max_speed_past_day" : data_x[i]["Max_speed_past_day"],
                    "richardson_formula_degrees_phi_2_solar_wind" : data_x[i]["richardson_formula_degrees_phi_2_solar_wind"],
                    "V log V" : data_x[i]["V log V"],
                    "halo" : data_x[i]["halo"],
                    "diffusive_shock" : data_x[i]["diffusive_shock"],
                    "Double_CME_100_MeV" : data_x[i]["Double_CME_100_MeV"],
                    "connection_angle_degrees" : data_x[i]["connection_angle_degrees"],
                    "connection_angle_degrees_phi_2_solar_wind_sq_div" : data_x[i]["connection_angle_degrees_phi_2_solar_wind_sq_div"],
                    "target" : target_y[i],
                    "expected_richardson" : data_x[i]["expected_richardson"],
                    "expected_richardson_ln" : data_x[i]["expected_richardson_ln"],
                    "100MeV_peak_intensity" : data_x[i]["100MeV_peak_intensity"],
                    "100MeV_peak_intensity_ln": data_y[i][0],
                    "predicted_100MeV_peak_intensity_ln" : pred_y[i][0]
                    }
                writer.writerow(row)
        return

    def getExpectedRichardson(self, indexes):
        real_data = get_real_data(self.orig_data, indexes)

        numElements = len(real_data)
        inputData_x = np.empty((numElements, 1), dtype=np.float64)
        inputData_y = np.empty((numElements, 1), dtype=np.float64)
    
        for i in range(0, numElements):
            elem = real_data[i]
        
            inputData_x[i][0] = float(elem["expected_richardson_ln"])
        
            inputData_y[i][0] = float(elem["100MeV_peak_intensity_ln"])
        return (inputData_x, inputData_y)

    def richardsonOriginal(self):
        testData_target = self.reg.getTargets("../res/gen/syn/firstStageTestSyn.csv")
        idxes = get_indexes("../res/gen/syn/firstStageTestSyn.csv")
        expected_richardson_x, one_hundred_mev_y = self.getExpectedRichardson(idxes)

        minXRichardson, maxXRichardson, minYRichardson, maxYRichardson = self.plotEvalGraph(one_hundred_mev_y, expected_richardson_x, testData_target, "../eval/expected_richardson/ExpectedRichardson.png", plotTitle="Richardson Formula MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Richardson Formula MeV Peak Intensity LN")

        for group in range(0, 3):
            testData_target = self.reg.getTargets("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))
            idxes = get_indexes("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))
            expected_richardson_x, one_hundred_mev_y = self.getExpectedRichardson(idxes)

            newMaxes = self.plotEvalGraph(one_hundred_mev_y, expected_richardson_x, testData_target, "../eval/expected_richardson/ExpectedRichardson_{}.png".format(group), plotTitle="Richardson Formula MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Richardson Formula MeV Peak Intensity LN")
            if newMaxes[0] < minXRichardson:
                minXRichardson = newMaxes[0]
            if newMaxes[1] > maxXRichardson:
                maxXRichardson = newMaxes[1]
            if newMaxes[2] < minYRichardson:
                minYRichardson = newMaxes[2]
            if newMaxes[3] > maxYRichardson:
                maxYRichardson = newMaxes[3]

        return (minXRichardson, maxXRichardson, minYRichardson, maxYRichardson)

    def richardsonOriginalScaled(self, overallMinX, overallMaxX, overallMinY, overallMaxY):
        testData_target = self.reg.getTargets("../res/gen/syn/firstStageTestSyn.csv")
        idxes = get_indexes("../res/gen/syn/firstStageTestSyn.csv")
        expected_richardson_x, one_hundred_mev_y = self.getExpectedRichardson(idxes)

        self.plotEvalGraphScaled(one_hundred_mev_y, expected_richardson_x, testData_target, "../eval/expected_richardson/ExpectedRichardsonScaled.png", overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Original LN Richardson", xLabel="100 MeV Peak Intensity LN", yLabel="Richardson Formula MeV Peak Intensity LN")

        for group in range(0, 3):
            testData_target = self.reg.getTargets("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))
            idxes = get_indexes("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))
            expected_richardson_x, one_hundred_mev_y = self.getExpectedRichardson(idxes)

            self.plotEvalGraphScaled(one_hundred_mev_y, expected_richardson_x, testData_target, "../eval/expected_richardson/ExpectedRichardson_{}_Scaled.png".format(group), overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Original LN Richardson Dataset {}".format(group + 1), xLabel="100 MeV Peak Intensity LN", yLabel="Richardson Formula MeV Peak Intensity LN")

    def learnRichardsonOnSyn(self):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/firstStageAllTrainingSyn.csv", useSynData=False)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn/adam_model_001_1_reg_init.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_on_syn/adam_model_001_1_reg_init.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

        print("\n\nFINAL LAYER WEIGHTS")
        layerNum = 0
        for layer in adam_model_001_1_retrained.layers:
            weights = layer.get_weights() # list of numpy arrays
            print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
            layerNum = layerNum + 1
        print("\n\n")

        keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_on_syn/model.png", show_shapes=True)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/firstStageTestSyn.csv", useSynData=False)
        testData_target = self.reg.getTargets("../res/gen/syn/firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)
        corr_richardson, mae_richardson = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
        corr_richardson_with_exp, mae_richardson_with_exp = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
        minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn/Predictions_Reg_NN.png", plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

        indexes = get_indexes("../res/gen/syn/firstStageTestSyn.csv")
        real_data = get_real_data(self.orig_data, indexes)
        self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn/Predictions_Reg_NN.csv")

        print("LEARN RICHARDSON ON SYN METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(corr_richardson, mae_richardson))
        print("learned,test,EXP,{},{}".format(corr_richardson_with_exp, mae_richardson_with_exp))
        return (minX, maxX, minY, maxY)

    def learnRichardsonOnSynScaled(self, overallMinX, overallMaxX, overallMinY, overallMaxY):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/firstStageAllTrainingSyn.csv", useSynData=False)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn/adam_model_001_1_reg_init.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_on_syn/adam_model_001_1_reg_init.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/firstStageTestSyn.csv")
        testData_target = self.reg.getTargets("../res/gen/syn/firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)

        self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn/Predictions_Reg_NN_Scaled.png", overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Scaled Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")
        return

    def learnRichardsonOnSynLong(self):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/firstStageAllTrainingSyn.csv", useSynData=False)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn/adam_model_001_1_reg_init_long.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_on_syn/adam_model_001_1_reg_init_long.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

        print("\n\nFINAL LAYER WEIGHTS")
        layerNum = 0
        for layer in adam_model_001_1_retrained.layers:
            weights = layer.get_weights() # list of numpy arrays
            print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
            layerNum = layerNum + 1
        print("\n\n")

        keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_on_syn/long_model.png", show_shapes=True)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/firstStageTestSyn.csv", useSynData=False)
        testData_target = self.reg.getTargets("../res/gen/syn/firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)
        corr_richardson, mae_richardson = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
        corr_richardson_with_exp, mae_richardson_with_exp = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
        minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn/Predictions_Reg_NN_Long.png", plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

        indexes = get_indexes("../res/gen/syn/firstStageTestSyn.csv")
        real_data = get_real_data(self.orig_data, indexes)
        self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn/Predictions_Reg_NN_Long.csv")

        print("LEARN RICHARDSON ON SYN LONG METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(corr_richardson, mae_richardson))
        print("learned,test,EXP,{},{}".format(corr_richardson_with_exp, mae_richardson_with_exp))
        return (minX, maxX, minY, maxY)

    def learnRichardsonOnSynLongScaled(self, overallMinX, overallMaxX, overallMinY, overallMaxY):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/firstStageAllTrainingSyn.csv", useSynData=False)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn/adam_model_001_1_reg_init_long.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_on_syn/adam_model_001_1_reg_init_long.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/firstStageTestSyn.csv", useSynData=False)
        testData_target = self.reg.getTargets("../res/gen/syn/firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)

        self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn/Predictions_Reg_NN_Long_Scaled.png", overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Scaled Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")
        return

    def learnRichardsonOnSynNoDoubleCME(self):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageAllTrainingSyn.csv", useSynData=False)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

        print("\n\nFINAL LAYER WEIGHTS")
        layerNum = 0
        for layer in adam_model_001_1_retrained.layers:
            weights = layer.get_weights() # list of numpy arrays
            print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
            layerNum = layerNum + 1
        print("\n\n")

        keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_on_syn_no_double_cme/model.png", show_shapes=True)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageTestSyn.csv", useSynData=False)
        testData_target = self.reg.getTargets("../res/gen/syn/no_double_cme_firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)
        corr_richardson, mae_richardson = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
        corr_richardson_with_exp, mae_richardson_with_exp = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
        minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_no_double_cme/Predictions_Reg_NN.png", plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

        indexes = get_indexes("../res/gen/syn/no_double_cme_firstStageTestSyn.csv")
        real_data = get_real_data(self.orig_data, indexes)
        self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_no_double_cme/Predictions_Reg_NN.csv")

        print("LEARN RICHARDSON ON SYN NO DOUBLE CME METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(corr_richardson, mae_richardson))
        print("learned,test,EXP,{},{}".format(corr_richardson_with_exp, mae_richardson_with_exp))
        return (minX, maxX, minY, maxY)

    def learnRichardsonOnSynNoDoubleCMEScaled(self, overallMinX, overallMaxX, overallMinY, overallMaxY):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageAllTrainingSyn.csv", useSynData=False)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageTestSyn.csv", useSynData=False)
        testData_target = self.reg.getTargets("../res/gen/syn/no_double_cme_firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)

        self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_no_double_cme/Predictions_Reg_NN_Scaled.png", overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Scaled Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")
        return

    def learnRichardsonOnSynNoDoubleCMELong(self):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageAllTrainingSyn.csv", useSynData=False)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init_long.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init_long.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

        print("\n\nFINAL LAYER WEIGHTS")
        layerNum = 0
        for layer in adam_model_001_1_retrained.layers:
            weights = layer.get_weights() # list of numpy arrays
            print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
            layerNum = layerNum + 1
        print("\n\n")

        keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_on_syn_no_double_cme/model_long.png", show_shapes=True)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageTestSyn.csv", useSynData=False)
        testData_target = self.reg.getTargets("../res/gen/syn/no_double_cme_firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)
        corr_richardson, mae_richardson = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
        corr_richardson_with_exp, mae_richardson_with_exp = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
        minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_no_double_cme/Predictions_Reg_NN_long.png", plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

        indexes = get_indexes("../res/gen/syn/firstStageTestSyn.csv")
        real_data = get_real_data(self.orig_data, indexes)
        self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_no_double_cme/Predictions_Reg_NN_long.csv")

        print("LEARN RICHARDSON ON SYN NO DOUBLE CME Long METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(corr_richardson, mae_richardson))
        print("learned,test,EXP,{},{}".format(corr_richardson_with_exp, mae_richardson_with_exp))
        return (minX, maxX, minY, maxY)

    def learnRichardsonOnSynNoDoubleCMELongScaled(self, overallMinX, overallMaxX, overallMinY, overallMaxY):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageAllTrainingSyn.csv", useSynData=False)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init_long.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_on_syn_no_double_cme/adam_model_001_1_reg_init_long.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/syn/no_double_cme_firstStageTestSyn.csv", useSynData=False)
        testData_target = self.reg.getTargets("../res/gen/syn/no_double_cme_firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)

        self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_no_double_cme/Predictions_Reg_NN_Scaled_long.png", overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Scaled Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")
        return

    def learnRichardsonSyn(self):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/firstStageAllTrainingSyn.csv", useSynData=True)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_syn/adam_model_001_1_reg_init.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_syn/adam_model_001_1_reg_init.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

        print("\n\nFINAL LAYER WEIGHTS")
        layerNum = 0
        for layer in adam_model_001_1_retrained.layers:
            weights = layer.get_weights() # list of numpy arrays
            print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
            layerNum = layerNum + 1
        print("\n\n")

        keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_syn/model.png", show_shapes=True)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/firstStageTestSyn.csv", useSynData=True)
        testData_target = self.reg.getTargets("../res/firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)
        corr_richardson, mae_richardson = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
        corr_richardson_with_exp, mae_richardson_with_exp = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
        minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_syn/Predictions_Reg_NN.png", plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

        indexes = get_indexes("../res/firstStageTestSyn.csv")
        real_data = get_real_data(self.orig_data, indexes)
        self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_syn/Predictions_Reg_NN.csv")

        print("LEARN RICHARDSON SYN METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(corr_richardson, mae_richardson))
        print("learned,test,EXP,{},{}".format(corr_richardson_with_exp, mae_richardson_with_exp))
        return (minX, maxX, minY, maxY)

    def learnRichardsonSynNoDoubleCME(self):
        richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/no_double_cme_firstStageAllTrainingSyn.csv", useSynData=True)

        adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_syn_no_double_cme/adam_model_001_1_reg_init.ckpt"

        adamLearningRate = 0.0001
        adamEpsilon = 1.0
        logFilename = "../out/retrained_learn_richardson_syn_no_double_cme/adam_model_001_1_reg_init.csv"

        retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
        adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

        print("\n\nFINAL LAYER WEIGHTS")
        layerNum = 0
        for layer in adam_model_001_1_retrained.layers:
            weights = layer.get_weights() # list of numpy arrays
            print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
            layerNum = layerNum + 1
        print("\n\n")

        keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_syn_no_double_cme/model.png", show_shapes=True)

        # Evaluate
        testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/no_double_cme_firstStageTestSyn.csv", useSynData=True)
        testData_target = self.reg.getTargets("../res/no_double_cme_firstStageTestSyn.csv")

        predictedData_y = adam_model_001_1_retrained.predict(testData_x)
        corr_richardson, mae_richardson = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
        corr_richardson_with_exp, mae_richardson_with_exp = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
        minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_syn_no_double_cme/Predictions_Reg_NN.png", plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

        indexes = get_indexes("../res/no_double_cme_firstStageTestSyn.csv")
        real_data = get_real_data(self.orig_data, indexes)
        self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_syn_no_double_cme/Predictions_Reg_NN.csv")

        print("LEARN RICHARDSON SYN NO DOUBLE CME METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(corr_richardson, mae_richardson))
        print("learned,test,EXP,{},{}".format(corr_richardson_with_exp, mae_richardson_with_exp))
        return (minX, maxX, minY, maxY)

    def learnRichardsonSyn3Fold(self):
        accumulated_corr_ln = 0.0
        accumulated_corr_intensity = 0.0
        accumulated_mae_ln = 0.0
        accumulated_mae_intensity = 0.0

        overall_minX = None
        overall_maxX = None
        overall_minY = None
        overall_maxY = None

        numFolds = 3
        dataset_weights = { 'w_v': 0, 'w_exp': 0 }
        for i in range(0, numFolds):

            adamLearningRate = 0.0001
            adamEpsilon = 1.0
            group = i

            richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageAllTraining_Syn_3Fold_{}.csv".format(group), useSynData=False)

            adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_3fold/group_{}_lR_{:.5f}.ckpt".format(group, adamLearningRate)

            logFilename = "../out/retrained_learn_richardson_on_syn_3fold/group_{}_lR_{:.5f}.csv".format(group, adamLearningRate)

            retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
            adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

            print("\n\nFINAL LAYER WEIGHTS: {}".format(i))
            layerNum = 0
            for layer in adam_model_001_1_retrained.layers:
                weights = layer.get_weights() # list of numpy arrays
                print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
                layerNum = layerNum + 1
            print("\n\n")

            layer0Weights = adam_model_001_1_retrained.layers[1].get_weights()
            w_v = layer0Weights[0][0][0]
            dataset_weights['w_v'] = dataset_weights['w_v'] + w_v
            w_0 = layer0Weights[1][0]
            w_exp = math.exp(w_0)
            print("\n\n\nW_0: {}. {}\n\n\n".format(w_0, w_exp))
            dataset_weights['w_exp'] = dataset_weights['w_exp'] + w_exp



            keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_on_syn_3fold/model.png", show_shapes=True)

            # Evaluate
            testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group), useSynData=False)
            testData_target = self.reg.getTargets("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))

            predictedData_y = adam_model_001_1_retrained.predict(testData_x)
            corr_ln, mae_ln = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
            corr_intensity, mae_intensity = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
            minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold/Predictions_Reg_NN_{}.png".format(group), plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

            if overall_minX is None:
                overall_minX = minX
            elif minX < overall_minX:
                overall_minX = minX
   
            if overall_maxX is None:
                overall_maxX = maxX
            elif maxX > overall_maxX:
                overall_maxX = maxX
            
            if overall_minY is None:
                overall_minY = minY
            elif minY < overall_minY:
                overall_minY = minY

            if overall_maxY is None:
                overall_maxY = maxY
            elif maxY > overall_maxY:
                overall_maxY = maxY

            indexes = get_indexes("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))
            real_data = get_real_data(self.orig_data, indexes)
            self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold/Predictions_Reg_NN_{}.csv".format(group))

            accumulated_corr_ln = accumulated_corr_ln + corr_ln
            accumulated_corr_intensity = accumulated_corr_intensity + corr_intensity
            accumulated_mae_ln = accumulated_mae_ln + mae_ln
            accumulated_mae_intensity = accumulated_mae_intensity + mae_intensity

        for key in dataset_weights.keys():
            dataset_weights[key] = dataset_weights[key] / numFolds

        accumulated_corr_ln = accumulated_corr_ln / numFolds
        accumulated_corr_intensity = accumulated_corr_intensity / numFolds
        accumulated_mae_ln = accumulated_mae_ln / numFolds
        accumulated_mae_intensity = accumulated_mae_intensity / numFolds

        print("LEARN RICHARDSON ON SYN 3FOLD METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(accumulated_corr_ln, accumulated_mae_ln))
        print("learned,test,INT,{},{}".format(accumulated_corr_intensity, accumulated_mae_intensity))
        return (overall_minX, overall_maxX, overall_minY, overall_maxY, dataset_weights)

    def learnRichardsonOnSyn3FoldScaled(self, overallMinX, overallMaxX, overallMinY, overallMaxY):

        numFolds = 3
        for i in range(0, numFolds):

            adamLearningRate = 0.0001
            adamEpsilon = 1.0
            group = i

            richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageAllTraining_Syn_3Fold_{}.csv".format(group), useSynData=False)

            adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_3fold/group_{}_lR_{:.5f}.ckpt".format(group, adamLearningRate)

            logFilename = "../out/retrained_learn_richardson_on_syn_3fold/group_{}_lR_{:.5f}.csv".format(group, adamLearningRate)

            retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
            adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

            # Evaluate
            testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group), useSynData=False)
            testData_target = self.reg.getTargets("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))

            predictedData_y = adam_model_001_1_retrained.predict(testData_x)
            
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold/Predictions_Reg_NN_Scaled_{}.png".format(group), overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Trained LN Richardson Dataset {}".format(group + 1), xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

            sep_only_testData_y = []
            sep_only_predictedData_y = []
            sep_only_testDataTarget = []
            for i in range(0, len(testData_target)):
                if testData_target[i] == 1:
                    sep_only_testData_y.append(testData_y[i])        
                    sep_only_predictedData_y.append(predictedData_y[i])
                    sep_only_testDataTarget.append(1)

            self.plotEvalGraphScaled(sep_only_testData_y, sep_only_predictedData_y, sep_only_testDataTarget, "../eval/retrained_learn_richardson_on_syn_3fold/Predictions_Reg_NN_Scaled_SEP_ONLY_{}.png".format(group), overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Trained LN Richardson Dataset {}".format(group + 1), xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")
        return

    def learnRichardsonSyn3FoldLong(self):
        accumulated_corr_ln = 0.0
        accumulated_corr_intensity = 0.0
        accumulated_mae_ln = 0.0
        accumulated_mae_intensity = 0.0

        overall_minX = None
        overall_maxX = None
        overall_minY = None
        overall_maxY = None

        numFolds = 3
        for i in range(0, numFolds):

            adamLearningRate = 0.0001
            adamEpsilon = 1.0
            group = i

            richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageAllTraining_Syn_3Fold_{}.csv".format(group), useSynData=False)

            adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_3fold/long_group_{}_lR_{:.5f}.ckpt".format(group, adamLearningRate)

            logFilename = "../out/retrained_learn_richardson_on_syn_3fold/long_group_{}_lR_{:.5f}.csv".format(group, adamLearningRate)

            retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
            adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

            print("\n\nFINAL LAYER WEIGHTS: {}".format(i))
            layerNum = 0
            for layer in adam_model_001_1_retrained.layers:
                weights = layer.get_weights() # list of numpy arrays
                print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
                layerNum = layerNum + 1
            print("\n\n")

            keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_on_syn_3fold/long_model.png", show_shapes=True)

            # Evaluate
            testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group), useSynData=False)
            testData_target = self.reg.getTargets("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))

            predictedData_y = adam_model_001_1_retrained.predict(testData_x)
            corr_ln, mae_ln = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
            corr_intensity, mae_intensity = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
            minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold/Predictions_Reg_NN_Long_{}.png".format(group), plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

            if overall_minX is None:
                overall_minX = minX
            elif minX < overall_minX:
                overall_minX = minX
   
            if overall_maxX is None:
                overall_maxX = maxX
            elif maxX > overall_maxX:
                overall_maxX = maxX
            
            if overall_minY is None:
                overall_minY = minY
            elif minY < overall_minY:
                overall_minY = minY

            if overall_maxY is None:
                overall_maxY = maxY
            elif maxY > overall_maxY:
                overall_maxY = maxY

            indexes = get_indexes("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))
            real_data = get_real_data(self.orig_data, indexes)
            self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold/Predictions_Reg_NN_Long_{}.csv".format(group))

            accumulated_corr_ln = accumulated_corr_ln + corr_ln
            accumulated_corr_intensity = accumulated_corr_intensity + corr_intensity
            accumulated_mae_ln = accumulated_mae_ln + mae_ln
            accumulated_mae_intensity = accumulated_mae_intensity + mae_intensity

        accumulated_corr_ln = accumulated_corr_ln / numFolds
        accumulated_corr_intensity = accumulated_corr_intensity / numFolds
        accumulated_mae_ln = accumulated_mae_ln / numFolds
        accumulated_mae_intensity = accumulated_mae_intensity / numFolds

        print("LEARN RICHARDSON ON SYN 3FOLD Long METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(accumulated_corr_ln, accumulated_mae_ln))
        print("learned,test,INT,{},{}".format(accumulated_corr_intensity, accumulated_mae_intensity))
        return (overall_minX, overall_maxX, overall_minY, overall_maxY)

    def learnRichardsonOnSyn3FoldLongScaled(self, overallMinX, overallMaxX, overallMinY, overallMaxY):
        numFolds = 3
        for i in range(0, numFolds):

            adamLearningRate = 0.0001
            adamEpsilon = 1.0
            group = i

            richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageAllTraining_Syn_3Fold_{}.csv".format(group), useSynData=False)

            adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_3fold/long_group_{}_lR_{:.5f}.ckpt".format(group, adamLearningRate)

            logFilename = "../out/retrained_learn_richardson_on_syn_3fold/long_group_{}_lR_{:.5f}.csv".format(group, adamLearningRate)

            retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
            adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

            # Evaluate
            testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group), useSynData=False)
            testData_target = self.reg.getTargets("../res/gen/fold/syn/firstStageTest_3Fold_Syn_{}.csv".format(group))

            predictedData_y = adam_model_001_1_retrained.predict(testData_x)
            
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold/Predictions_Reg_NN_Long_Scaled_{}.png".format(group), overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Scaled Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

        return

    def learnRichardsonSyn3FoldNoDoubleCME(self):
        accumulated_corr_ln = 0.0
        accumulated_corr_intensity = 0.0
        accumulated_mae_ln = 0.0
        accumulated_mae_intensity = 0.0

        overall_minX = None
        overall_maxX = None
        overall_minY = None
        overall_maxY = None

        numFolds = 3
        for i in range(0, numFolds):

            adamLearningRate = 0.0001
            adamEpsilon = 1.0
            group = i

            richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/no_double_cme_firstStageAllTraining_Syn_3Fold_{}.csv".format(group), useSynData=False)

            adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_3fold_no_double_cme/group_{}_lR_{:.5f}.ckpt".format(group, adamLearningRate)

            logFilename = "../out/retrained_learn_richardson_on_syn_3fold_no_double_cme/group_{}_lR_{:.5f}.csv".format(group, adamLearningRate)

            retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
            adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

            print("\n\nFINAL LAYER WEIGHTS: {}".format(i))
            layerNum = 0
            for layer in adam_model_001_1_retrained.layers:
                weights = layer.get_weights() # list of numpy arrays
                print("LAYER: {}. WEIGHTS: {}".format(layerNum, weights))
                layerNum = layerNum + 1
            print("\n\n")

            keras.utils.plot_model(adam_model_001_1_retrained, to_file="../eval/retrained_learn_richardson_on_syn_3fold_no_double_cme/model.png", show_shapes=True)

            # Evaluate
            testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/no_double_cme_firstStageTest_3Fold_Syn_{}.csv".format(group), useSynData=False)
            testData_target = self.reg.getTargets("../res/gen/fold/syn/no_double_cme_firstStageTest_3Fold_Syn_{}.csv".format(group))

            predictedData_y = adam_model_001_1_retrained.predict(testData_x)
            corr_ln, mae_ln = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=False)
            corr_intensity, mae_intensity = self.evaluate(testData_y, predictedData_y, testData_target, exp_before_mae=True)
            minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold_no_double_cme/Predictions_Reg_NN_{}.png".format(group), plotTitle="Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

            if overall_minX is None:
                overall_minX = minX
            elif minX < overall_minX:
                overall_minX = minX
   
            if overall_maxX is None:
                overall_maxX = maxX
            elif maxX > overall_maxX:
                overall_maxX = maxX
            
            if overall_minY is None:
                overall_minY = minY
            elif minY < overall_minY:
                overall_minY = minY

            if overall_maxY is None:
                overall_maxY = maxY
            elif maxY > overall_maxY:
                overall_maxY = maxY

            indexes = get_indexes("../res/gen/fold/syn/no_double_cme_firstStageTest_3Fold_Syn_{}.csv".format(group))
            real_data = get_real_data(self.orig_data, indexes)
            self.outputPredictions(real_data, testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold_no_double_cme/Predictions_Reg_NN_{}.csv".format(group))

            accumulated_corr_ln = accumulated_corr_ln + corr_ln
            accumulated_corr_intensity = accumulated_corr_intensity + corr_intensity
            accumulated_mae_ln = accumulated_mae_ln + mae_ln
            accumulated_mae_intensity = accumulated_mae_intensity + mae_intensity

        accumulated_corr_ln = accumulated_corr_ln / numFolds
        accumulated_corr_intensity = accumulated_corr_intensity / numFolds
        accumulated_mae_ln = accumulated_mae_ln / numFolds
        accumulated_mae_intensity = accumulated_mae_intensity / numFolds

        print("LEARN RICHARDSON ON SYN 3FOLD No Double CME METRICS")
        print("approach,dataset,maeType,corr,mae")
        print("learned,test,LN,{},{}".format(accumulated_corr_ln, accumulated_mae_ln))
        print("learned,test,INT,{},{}".format(accumulated_corr_intensity, accumulated_mae_intensity))
        return (overall_minX, overall_maxX, overall_minY, overall_maxY)

    def learnRichardsonOnSyn3FoldNoDoubleCMEScaled(self, overallMinX, overallMaxX, overallMinY, overallMaxY):
        numFolds = 3
        for i in range(0, numFolds):

            adamLearningRate = 0.0001
            adamEpsilon = 1.0
            group = i

            richardson_trainingData_x, richardson_trainingData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/firstStageAllTraining_Syn_3Fold_{}.csv".format(group), useSynData=False)

            adam_model_001_1_retrained_checkpoint_path = "../out/retrained_learn_richardson_on_syn_3fold_no_double_cme/group_{}_lR_{:.5f}.ckpt".format(group, adamLearningRate)

            logFilename = "../out/retrained_learn_richardson_on_syn_3fold_no_double_cme/group_{}_lR_{:.5f}.csv".format(group, adamLearningRate)

            retrained_weights_filename = adam_model_001_1_retrained_checkpoint_path
            adam_model_001_1_retrained, adam_history_001_1_retrained = self.reg.trainRichardson(adamLearningRate, richardson_trainingData_x, richardson_trainingData_y, logFilename, adamEpsilon=adamEpsilon, weights_filename=retrained_weights_filename)

            # Evaluate
            testData_x, testData_y  = self.reg.getFirstStageInputRichardson("../res/gen/fold/syn/no_double_cme_firstStageTest_3Fold_Syn_{}.csv".format(group), useSynData=False)
            testData_target = self.reg.getTargets("../res/gen/fold/syn/no_double_cme_firstStageTest_3Fold_Syn_{}.csv".format(group))

            predictedData_y = adam_model_001_1_retrained.predict(testData_x)
            
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "../eval/retrained_learn_richardson_on_syn_3fold_no_double_cme/Predictions_Reg_NN_Scaled_{}.png".format(group), overallMinX, overallMaxX, overallMinY, overallMaxY, plotTitle="Scaled Predicted 100 MeV Peak Intensity LN vs 100 MeV Peak Intensity LN", xLabel="100 MeV Peak Intensity LN", yLabel="Predicted 100 MeV Peak Intensity LN")

        return

if __name__ == "__main__":
    evaluator = EvalRichardson()
    #minXOnSyn, maxXOnSyn, minYOnSyn, maxYOnSyn = evaluator.learnRichardsonOnSyn()
    #minXOnSynLong, maxXOnSynLong, minYOnSynLong, maxYOnSynLong = evaluator.learnRichardsonOnSynLong()

    minXOnSyn3Fold, maxXOnSyn3Fold, minYOnSyn3Fold, maxYOnSyn3Fold, dataset_weights = evaluator.learnRichardsonSyn3Fold()
    #minXOnSyn3FoldLong, maxXOnSyn3FoldLong, minYOnSyn3FoldLong, maxYOnSyn3FoldLong = evaluator.learnRichardsonSyn3FoldLong()

    #minXOnSyn3FoldNoDCMNE, maxXOnSyn3FoldNoDCMNE, minYOnSyn3FoldNoDCMNE, maxYOnSyn3FoldNoDCMNE = evaluator.learnRichardsonSyn3FoldNoDoubleCME()

    #minXOnSynNoDCMNE, maxXOnSynNoDCMNE, minYOnSynNoDCMNE, maxYOnSynNoDCMNE = evaluator.learnRichardsonOnSynNoDoubleCME()
    #minXOnSynNoDCMNELong, maxXOnSynNoDCMNELong, minYOnSynNoDCMNELong, maxYOnSynNoDCMNELong = evaluator.learnRichardsonOnSynNoDoubleCMELong()
    
    minXRichardson, maxXRichardson, minYRichardson, maxYRichardson = evaluator.richardsonOriginal()

    overallMinX = min((minXOnSyn3Fold, minXRichardson))
    overallMaxX = max((maxXOnSyn3Fold, maxXRichardson))
    overallMinY = min((minYOnSyn3Fold, minYRichardson))
    overallMaxY = max((maxYOnSyn3Fold, maxYRichardson))
    
    #evaluator.learnRichardsonOnSynScaled(overallMinX, overallMaxX, overallMinY, overallMaxY)
    #evaluator.learnRichardsonOnSynLongScaled(overallMinX, overallMaxX, overallMinY, overallMaxY)
    
    evaluator.learnRichardsonOnSyn3FoldScaled(overallMinX, overallMaxX, overallMinY, overallMaxY)
    #evaluator.learnRichardsonOnSyn3FoldNoDoubleCMEScaled(overallMinX, overallMaxX, overallMinY, overallMaxY)
    
    #evaluator.learnRichardsonOnSyn3FoldLongScaled(overallMinX, overallMaxX, overallMinY, overallMaxY)
    #evaluator.learnRichardsonOnSynNoDoubleCMEScaled(overallMinX, overallMaxX, overallMinY, overallMaxY)
    #evaluator.learnRichardsonOnSynNoDoubleCMELongScaled(overallMinX, overallMaxX, overallMinY, overallMaxY)
    
    evaluator.richardsonOriginalScaled(overallMinX, overallMaxX, overallMinY, overallMaxY)
    
    #evaluator.learnRichardsonSyn()
    #evaluator.learnRichardsonSynNoDoubleCME()

    fieldnames = []
    for key in dataset_weights.keys():
        print("KEY: {}. VAL: {}".format(key, dataset_weights[key]))
        fieldnames.append(key)

    csvFilename = "../res/trained/richardson.csv"
    with open(csvFilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(dataset_weights)

