from eval_regression import *
from results_dense import *

class EvalDenseLoss(EvalRegression):
    def __init__(self):
        super().__init__()
        self.results = ResultsDense()

    def denseLoss(self, testDataFilename, adamLearningRate, adamEpsilon, alpha):
        resFolder = "../res/gen"
        outFolder = "../out/denseLoss"
        parentEvalFolder = "../eval/denseLoss"

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

        for i in range(0, len(alphas)):
            dense_alpha = alphas[i]
            evalFolder = parentEvalFolder + "/alpha_{:.2f}".format(dense_alpha)

            weights_filename = "{}/alpha_{:.2f}.ckpt".format(outFolder, dense_alpha)
            print("WEIGHTS_FILENAME: {}".format(weights_filename))

            adam_model_001_1, _ = self.reg.denseLossNNModel(adamLearningRate, None, None, None, seed=1234, denseLossAlpha=dense_alpha, adamEpsilon=adamEpsilon, alpha=alpha, weights_filename=weights_filename)
            predictedData_y = adam_model_001_1.predict(testData_x)
            reg_corr, reg_mae, reg_FP, reg_FN, reg_TP, reg_TN, reg_F1, reg_HSS, reg_TSS = self.evaluate(testData_y, predictedData_y, testData_target)

            self.results.add_all_metric("dense", dense_alpha, {
                "corr": reg_corr,
                "mae": reg_mae,
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

        self.results.updateMinX(min_x)
        self.results.updateMaxX(max_x)
        self.results.updateMinY(min_y)
        self.results.updateMaxY(max_y)


    
if __name__ == "__main__":
    evaluator = EvalDenseLoss()
    #evaluator.regNN()
    #evaluator.regNN_iterate()

    #for i in range(1, 10):
    #    oversample_amount = 10 * i

    #    adamLearningRate = 0.001
    #    alpha = 0.3
    #    evaluator.regNN_oversampled(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(i), adamLearningRate, alpha=alpha)

    #    adamLearningRate = 0.001
    #    alpha = 0.3
    #    evaluator.rRT(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(i), adamLearningRate, alpha=alpha)

    #    adamLearningRate = 0.001
    #    evaluator.autoencoder(oversample_amount, "secondStageOversampleTest_percentSEP_0.{}.csv".format(i), adamLearningRate)

    evaluator.denseLoss("firstStageTest.csv", 0.001, 1.0, 0.3)

    evaluator.results.print_metrics()
    evaluator.results.print_F1()
