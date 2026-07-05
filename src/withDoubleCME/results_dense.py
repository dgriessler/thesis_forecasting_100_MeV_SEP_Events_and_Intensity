from results import *

def sort_F1_func(e):
    return e["F1"]

class ResultsDense(Results):
    def __init__(self):
        super().__init__()

        self.seenF1Scores["dense"] = dict()
        self.alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        for i in range(0, len(self.alphas)):
            self.seenF1Scores["dense"][self.alphas[i]] = dict()

        self.metrics["dense"] = dict()
        for i in range(0, len(self.alphas)):
            self.metrics["dense"][self.alphas[i]] = dict()

        self.all_metrics["dense"] = dict()
        for i in range(0, len(self.alphas)):
            self.all_metrics["dense"][self.alphas[i]] = list()

    def print_metrics(self):
        super().print_metrics()
        for i in range(0, len(self.alphas)):
            print(self.metrics_to_string("dense", self.alphas[i]))
