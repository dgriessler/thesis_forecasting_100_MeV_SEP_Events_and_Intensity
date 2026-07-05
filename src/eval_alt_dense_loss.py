from eval_regression import *
from results_dense import *
from regression_alt import *

class EvalAltDenseLoss(EvalRegression):
    def __init__(self):
        super().__init__()
        self.results = ResultsDense()
        self.reg = RegressionAlt()

    def denseLoss_iterate(self, testDataFilename, adamLearningRate, adamEpsilon, alpha):
        resFolder = "../res/gen"
        outFolder = "../out/alt_denseLoss_retrained"
        parentEvalFolder = "../eval/alt_denseLoss/iterate"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1ScoresArr = []

        min_x = None
        max_x = None
        min_y = None
        max_y = None

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

        self.results.updateMinX(min_x)
        self.results.updateMaxX(max_x)
        self.results.updateMinY(min_y)
        self.results.updateMaxY(max_y)

    def rRT(self, testDataFilename, adamLearningRate, adamEpsilon, alpha):
        resFolder = "../res/gen"
        outFolder = "../out/alt_denseLoss_retrained_rRT"
        parentEvalFolder = "../eval/alt_denseLoss_retrained_rRT"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1ScoresArr = []

        min_x = None
        max_x = None
        min_y = None
        max_y = None

        #alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
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

        self.results.updateMinX(min_x)
        self.results.updateMaxX(max_x)
        self.results.updateMinY(min_y)
        self.results.updateMaxY(max_y)

    def autoencoder(self, testDataFilename, adamLearningRate, adam_epsilon, alpha):
        resFolder = "../res/gen"
        outFolder = "../out/alt_denseLoss_retrained_autoencoder_ss"
        parentEvalFolder = "../eval/alt_denseLoss_retrained_autoencoder_ss"

        testData_x, testData_y  = self.reg.getFirstStageInput("{}/{}".format(resFolder, testDataFilename))
        testData_target = self.reg.getTargets("{}/{}".format(resFolder, testDataFilename))
        indexes = get_indexes("{}/{}".format(resFolder, testDataFilename))
        seenF1ScoresArr = []
        
        min_x = None
        max_x = None
        min_y = None
        max_y = None

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

                self.results.add_all_metric("dense_aut", dense_alpha, {
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


            self.results.updateMinX(min_x)
            self.results.updateMaxX(max_x)
            self.results.updateMinY(min_y)
            self.results.updateMaxY(max_y)

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

            self.results.updateMetrics("dense_aut", dense_alpha, {
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

if __name__ == "__main__":
    evaluator = EvalAltDenseLoss()

    adamLearningRate = 0.001
    adamEpsilon = 1.0
    alpha = 0.3
    #evaluator.denseLoss("firstStageTest.csv", adamLearningRate, adamEpsilon, alpha)
    #evaluator.denseLoss_iterate("firstStageTest.csv", adamLearningRate, adamEpsilon, alpha)

    adamLearningRate = 0.001
    adamEpsilon = 1.0
    alpha = 0.3
    evaluator.rRT("firstStageTest.csv", adamLearningRate, adamEpsilon, alpha=alpha)

    #adamLearningRate = 0.001
    #adamEpsilon = 1.0
    #alpha = 0.3
    #evaluator.autoencoder("firstStageTest.csv", adamLearningRate, adamEpsilon, alpha=alpha)

    evaluator.results.print_metrics()
    evaluator.results.print_F1()
