from eval_regression import *
from results_dense import *
from regression import *

class EvalDenseLoss(EvalRegression):
    def __init__(self):
        super().__init__()
        self.results = ResultsDense()
        self.reg = Regression()

    def denseLoss(self, testDataFilename, adamLearningRate, adamEpsilon, alpha):
        resFolder = "../res/gen"
        outFolder = "../out/denseLoss_retrained"
        parentEvalFolder = "../eval/denseLoss"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1ScoresArr = []

        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        for i in range(0, len(alphas)):
            seenF1ScoresArr.append(self.results.getSeenF1Scores("dense", alphas[i]))

        for i in range(0, len(seenF1ScoresArr)):
            seenF1ScoresArr[i].clear()

        for i in range(0, len(alphas)):
            dense_alpha = alphas[i]
            evalFolder = parentEvalFolder + "/alpha_{:.2f}".format(dense_alpha)

            weights_filename = "{}/alpha_{:.2f}.ckpt".format(outFolder, dense_alpha)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))

            adam_model_001_1, _ = self.reg.denseLossNNModel(adamLearningRate, None, None, None, seed=1234, denseLossAlpha=dense_alpha, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = adam_model_001_1.predict(testData_x)
            regNN_corr_sep, regNN_corr_sep_elevated, regNN_meanAbsoluteErrorSEP, regNN_meanAbsoluteErrorAll, reg_FP, reg_FN, reg_TP, reg_TN, reg_F1, reg_HSS, reg_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            self.results.add_all_metric("dense", dense_alpha, {
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

            evalFilename = getFilename("{}/Predictions_F1_".format(evalFolder), reg_F1, seenF1ScoresArr[i], trailer="")
            minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
            self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename))

    def denseLoss_iterate(self, testDataFilename, adamLearningRate, adamEpsilon, alpha):
        resFolder = "../res/gen"
        outFolder = "../out/denseLoss_retrained"
        parentEvalFolder = "../eval/denseLoss/iterate"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1ScoresArr = []

        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        for i in range(0, len(alphas)):
            seenF1ScoresArr.append(self.results.getSeenF1Scores("dense", alphas[i]))

        for i in range(0, len(seenF1ScoresArr)):
            seenF1ScoresArr[i].clear()

        numIterations = 5
        for i in range(0, len(alphas)):

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

            seenF1ScoresArr[i].clear()

            stats = []

            for j in range(0, numIterations):
                dense_alpha = alphas[i]
                evalFolder = parentEvalFolder + "/alpha_{:.2f}".format(dense_alpha)

                weights_filename = "{}/alpha_{:.2f}_it_{}.ckpt".format(outFolder, dense_alpha, j)
                print("WEIGHTS_FILENAME: {}".format(weights_filename))

                adam_model_001_1, _ = self.reg.denseLossNNModel(adamLearningRate, None, None, None, seed=1234+j, denseLossAlpha=dense_alpha, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)
                predictedData_y = adam_model_001_1.predict(testData_x)
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

                self.results.add_all_metric("dense", dense_alpha, {
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

                evalFilename = getFilename("{}/Predictions_F1_".format(evalFolder), reg_F1, seenF1ScoresArr[i], trailer="")
                minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
                self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename))

                stats.append({
                    "min_x": minX,
                    "max_x": maxX,
                    "min_y": minY,
                    "max_y": maxY,
                    "F1": reg_F1
                })

            sorted_stats = sorted(stats, key=lambda x: x["F1"])

            self.results.updateMinX("dense", alphas[i], sorted_stats[2]["min_x"][0])
            self.results.updateMaxX("dense", alphas[i], sorted_stats[2]["max_x"][0])
            self.results.updateMinY("dense", alphas[i], sorted_stats[2]["min_y"][0])
            self.results.updateMaxY("dense", alphas[i], sorted_stats[2]["max_y"][0])

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
            self.results.updateMetrics("dense", dense_alpha, {
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

    def denseLoss_iterate_scaled(self, dense_alpha, testDataFilename, adamLearningRate, adamEpsilon, alpha, min_x, max_x, min_y, max_y):
        resFolder = "../res/gen"
        outFolder = "../out/denseLoss_retrained"
        parentEvalFolder = "../eval/denseLoss/iterate"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))

        seenF1Scores = self.results.getSeenF1Scores("dense", dense_alpha)
        seenF1Scores.clear()

        numIterations = 5

        for j in range(0, numIterations):
            evalFolder = parentEvalFolder + "/alpha_{:.2f}".format(dense_alpha)

            weights_filename = "{}/alpha_{:.2f}_it_{}.ckpt".format(outFolder, dense_alpha, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))

            adam_model_001_1, _ = self.reg.denseLossNNModel(adamLearningRate, None, None, None, seed=1234+j, denseLossAlpha=dense_alpha, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = adam_model_001_1.predict(testData_x)
            regNN_corr_sep, regNN_corr_sep_elevated, regNN_meanAbsoluteErrorSEP, regNN_meanAbsoluteErrorAll, reg_FP, reg_FN, reg_TP, reg_TN, reg_F1, reg_HSS, reg_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            evalFilename = getFilename("{}/Scaled_Predictions_F1_".format(evalFolder), reg_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), min_x, max_x, min_y, max_y, plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")


    def rRT(self, testDataFilename, adamLearningRate, adamEpsilon, alpha):
        resFolder = "../res/gen"
        outFolder = "../out/denseLoss_retrained_rRT"
        parentEvalFolder = "../eval/denseLoss_retrained_rRT"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))

        seenF1ScoresArr = []

        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        for i in range(0, len(alphas)):
            seenF1ScoresArr.append(self.results.getSeenF1Scores("dense_rRT", alphas[i]))

        for i in range(0, len(seenF1ScoresArr)):
            seenF1ScoresArr[i].clear()

        numIterations = 5
        for i in range(0, len(alphas)):
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

            seenF1ScoresArr[i].clear()

            stats = []

            for j in range(0, numIterations):
                dense_alpha = alphas[i]
                evalFolder = parentEvalFolder + "/alpha_{:.2f}".format(dense_alpha)

                weights_filename = "{}/alpha_{:.2f}_it_{}.ckpt".format(outFolder, dense_alpha, j)
                print("WEIGHTS_FILENAME: {}".format(weights_filename))

                retrained_adamLearningRate = 0.001
                retrained_adamEpsilon = 1.0
                retrained_alpha = 0.3
                retrained_weights_filename = "../out/regNN_retrained/adam_model_001_1_retrained.ckpt"
                adam_model_001_1_retrained, _ = self.reg.regNN(retrained_adamLearningRate, None, None, None, 1234, retrained_adamEpsilon, retrained_alpha, weights_filename=retrained_weights_filename)

                feature_extractor = get_feature_extractor(adam_model_001_1_retrained)
                NNModel, _ = self.reg.denseLoss_rRT(adamLearningRate, feature_extractor, None, None, None, dense_alpha, seed=1234+j, weights_filename=weights_filename, adamEpsilon=adamEpsilon, alpha=alpha)
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

                self.results.add_all_metric("dense_rRT", dense_alpha, {
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

                evalFilename = getFilename("{}/Predictions_F1_".format(evalFolder), rRT_F1, seenF1ScoresArr[i], trailer="")
                minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
                self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename))

                stats.append({
                    "min_x": minX,
                    "max_x": maxX,
                    "min_y": minY,
                    "max_y": maxY,
                    "F1": rRT_F1
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
        
            self.results.updateMetrics("dense_rRT", dense_alpha, {
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

            sorted_stats = sorted(stats, key=lambda x: x["F1"])

            self.results.updateMinX("dense_rRT", alphas[i], sorted_stats[2]["min_x"][0])
            self.results.updateMaxX("dense_rRT", alphas[i], sorted_stats[2]["max_x"][0])
            self.results.updateMinY("dense_rRT", alphas[i], sorted_stats[2]["min_y"][0])
            self.results.updateMaxY("dense_rRT", alphas[i], sorted_stats[2]["max_y"][0])

    def rRT_scaled(self, dense_alpha, testDataFilename, adamLearningRate, adamEpsilon, alpha, min_x, max_x, min_y, max_y):
        resFolder = "../res/gen"
        outFolder = "../out/denseLoss_retrained_rRT"
        parentEvalFolder = "../eval/denseLoss_retrained_rRT"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))

        seenF1Scores = self.results.getSeenF1Scores("dense_rRT", dense_alpha)
        seenF1Scores.clear()

        numIterations = 5

        for j in range(0, numIterations):
            evalFolder = parentEvalFolder + "/alpha_{:.2f}".format(dense_alpha)

            weights_filename = "{}/alpha_{:.2f}_it_{}.ckpt".format(outFolder, dense_alpha, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))

            retrained_adamLearningRate = 0.001
            retrained_adamEpsilon = 1.0
            retrained_alpha = 0.3
            retrained_weights_filename = "../out/regNN_retrained/adam_model_001_1_retrained.ckpt"
            adam_model_001_1_retrained, _ = self.reg.regNN(retrained_adamLearningRate, None, None, None, 1234, retrained_adamEpsilon, retrained_alpha, weights_filename=retrained_weights_filename)

            feature_extractor = get_feature_extractor(adam_model_001_1_retrained)
            NNModel, _ = self.reg.denseLoss_rRT(adamLearningRate, feature_extractor, None, None, None, dense_alpha, seed=1234+j, weights_filename=weights_filename, adamEpsilon=adamEpsilon, alpha=alpha)
            predictedData_y = NNModel.predict(testData_x)
            rRT_corr_sep, rRT_corr_sep_elevated, rRT_meanAbsoluteErrorSEP, rRT_meanAbsoluteErrorAll, rRT_FP, rRT_FN, rRT_TP, rRT_TN, rRT_F1, rRT_HSS, rRT_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            evalFilename = getFilename("{}/Scaled_Predictions_F1_".format(evalFolder), rRT_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), min_x, max_x, min_y, max_y, plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")

    def autoencoder(self, testDataFilename, adamLearningRate, adam_epsilon, alpha):
        resFolder = "../res/gen"
        outFolder = "../out/denseLoss_retrained_autoencoder_ss"
        parentEvalFolder = "../eval/denseLoss_retrained_autoencoder_ss"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        
        seenF1ScoresArr = []

        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        for i in range(0, len(alphas)):
            seenF1ScoresArr.append(self.results.getSeenF1Scores("dense_aut", alphas[i]))

        for i in range(0, len(seenF1ScoresArr)):
            seenF1ScoresArr[i].clear()

        numIterations = 5
        for i in range(0, len(alphas)):
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

            seenF1ScoresArr[i].clear()

            stats = []

            for j in range(0, numIterations):
                dense_alpha = alphas[i]
                evalFolder = parentEvalFolder + "/alpha_{:.2f}".format(dense_alpha)

                weights_filename = "{}/alpha_{:.2f}_it_{}.ckpt".format(outFolder, dense_alpha, j)
                print("WEIGHTS_FILENAME: {}".format(weights_filename))

                autoencoder_validation_weights_path = "../out/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
                adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)
                adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)

                autoencoder_second_stage_model = self.reg.denseLoss_autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, None, None, None, None, dense_alpha, alpha, weights_filename=weights_filename, train=False, seed=1234+j)
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

                self.results.add_all_metric("dense_aut", alphas[i], {
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

                evalFilename = getFilename("{}/Predictions_F1_".format(evalFolder), aut_F1, seenF1ScoresArr[i], trailer="")
                minX, maxX, minY, maxY = self.plotEvalGraph(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")
                self.outputPredictions(get_real_data(self.orig_data, indexes), testData_y, predictedData_y, testData_target, "{}.csv".format(evalFilename))

                stats.append({
                    "min_x": minX,
                    "max_x": maxX,
                    "min_y": minY,
                    "max_y": maxY,
                    "F1": aut_F1
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

            self.results.updateMetrics("dense_aut", alphas[i], {
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

            sorted_stats = sorted(stats, key=lambda x: x["F1"])

            self.results.updateMinX("dense_aut", alphas[i], sorted_stats[2]["min_x"][0])
            self.results.updateMaxX("dense_aut", alphas[i], sorted_stats[2]["max_x"][0])
            self.results.updateMinY("dense_aut", alphas[i], sorted_stats[2]["min_y"][0])
            self.results.updateMaxY("dense_aut", alphas[i], sorted_stats[2]["max_y"][0])

    def autoencoder_scaled(self, dense_alpha, testDataFilename, adamLearningRate, adam_epsilon, alpha, min_x, max_x, min_y, max_y):
        resFolder = "../res/gen"
        outFolder = "../out/denseLoss_retrained_autoencoder_ss"
        parentEvalFolder = "../eval/denseLoss_retrained_autoencoder_ss"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        
        seenF1Scores = self.results.getSeenF1Scores("dense_aut", dense_alpha)
        seenF1Scores.clear()

        numIterations = 5

        for j in range(0, numIterations):
            evalFolder = parentEvalFolder + "/alpha_{:.2f}".format(dense_alpha)

            weights_filename = "{}/alpha_{:.2f}_it_{}.ckpt".format(outFolder, dense_alpha, j)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))

            autoencoder_validation_weights_path = "../out/autoencoder_retrained/autoencoder_validation_training_retrained.ckpt"
            adamOptimizer = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)
            adamOptimizerSS = keras.optimizers.Adam(learning_rate=adamLearningRate, epsilon=adam_epsilon)

            autoencoder_second_stage_model = self.reg.denseLoss_autoencoder_second_stage_all(autoencoder_validation_weights_path, adamOptimizer, adamOptimizerSS, None, None, None, None, dense_alpha, alpha, weights_filename=weights_filename, train=False, seed=1234+j)
            predictedData_y = autoencoder_second_stage_model.predict(testData_x)
            aut_corr_sep, aut_corr_sep_elevated, aut_meanAbsoluteErrorSEP, aut_meanAbsoluteErrorAll, aut_FP, aut_FN, aut_TP, aut_TN, aut_F1, aut_HSS, aut_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            evalFilename = getFilename("{}/Scaled_Predictions_F1_".format(evalFolder), aut_F1, seenF1Scores, trailer="")
            self.plotEvalGraphScaled(testData_y, predictedData_y, testData_target, "{}.png".format(evalFilename), min_x, max_x, min_y, max_y, plotTitle="Predicted Peak Intensity LN vs Original Peak Intensity LN", xLabel="Peak Intensity LN", yLabel="Predicted Peak Intensity LN")

if __name__ == "__main__":
    evaluator = EvalDenseLoss()

    adamLearningRate = 0.001
    adamEpsilon = 1.0
    alpha = 0.3
    #evaluator.denseLoss("firstStageTest.csv", adamLearningRate, adamEpsilon, alpha)
    evaluator.denseLoss_iterate("firstStageTest.csv", adamLearningRate, adamEpsilon, alpha)

    adamLearningRate = 0.001
    adamEpsilon = 1.0
    alpha = 0.3
    evaluator.rRT("firstStageTest.csv", adamLearningRate, adamEpsilon, alpha=alpha)

    adamLearningRate = 0.001
    adamEpsilon = 1.0
    alpha = 0.3
    evaluator.autoencoder("firstStageTest.csv", adamLearningRate, adamEpsilon, alpha=alpha)

    evaluator.results.print_metrics()
    evaluator.results.print_F1()

    min_x = min((-2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1)) - 0.1
    max_x = max((4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1)) + 0.1
    min_y = min((-3.0, -2.4, -2.4, -3.2, -2.1, -2.7, -4.8, -4.0, -4.3, -2.3, -2.5, -2.4)) - 0.1
    max_y = max((2.0, 0.7, 0.8, 1.8, 2.4, 1.1, 4.4, 1.7, 1.9, 1.1, 0.8, 0.6)) + 0.1

    #evaluator.denseLoss_iterate_scaled(0.9, "firstStageTest.csv", adamLearningRate, adamEpsilon, alpha, min_x, max_x, min_y, max_y)

    #evaluator.rRT_scaled(0.7, "firstStageTest.csv", adamLearningRate, adamEpsilon, alpha, min_x, max_x, min_y, max_y)

    evaluator.autoencoder_scaled(0.6, "firstStageTest.csv", adamLearningRate, adamEpsilon, alpha, min_x, max_x, min_y, max_y)

