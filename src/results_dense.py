from results import *

class ResultsDense(Results):
    def __init__(self):
        super().__init__()

        self.seenF1Scores["dense"] = dict()
        self.seenF1Scores["dense_rRT"] = dict()
        self.seenF1Scores["dense_aut"] = dict()
        self.alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        for i in range(0, len(self.alphas)):
            self.seenF1Scores["dense"][self.alphas[i]] = dict()
            self.seenF1Scores["dense_rRT"][self.alphas[i]] = dict()
            self.seenF1Scores["dense_aut"][self.alphas[i]] = dict()

        self.metrics["dense"] = dict()
        self.metrics["dense_rRT"] = dict()
        self.metrics["dense_aut"] = dict()
        for i in range(0, len(self.alphas)):
            self.metrics["dense"][self.alphas[i]] = dict()
            self.metrics["dense_rRT"][self.alphas[i]] = dict()
            self.metrics["dense_aut"][self.alphas[i]] = dict()

        self.all_metrics["dense"] = dict()
        self.all_metrics["dense_rRT"] = dict()
        self.all_metrics["dense_aut"] = dict()
        for i in range(0, len(self.alphas)):
            self.all_metrics["dense"][self.alphas[i]] = list()
            self.all_metrics["dense_rRT"][self.alphas[i]] = list()
            self.all_metrics["dense_aut"][self.alphas[i]] = list()

        self.stats["dense"] = dict()
        self.stats["dense_rRT"] = dict()
        self.stats["dense_aut"] = dict()
        for key in self.stats:
            for i in range(0, len(self.alphas)):
                self.stats[key][self.alphas[i]] = dict()
                self.stats[key][self.alphas[i]]["min_x"] = None
                self.stats[key][self.alphas[i]]["max_x"] = None
                self.stats[key][self.alphas[i]]["min_y"] = None
                self.stats[key][self.alphas[i]]["max_y"] = None

    def updateMinX(self, t, alpha, min_x):
        if self.stats[t][alpha]["min_x"] is None:
            self.stats[t][alpha]["min_x"] = min_x
        else:
            self.stats[t][alpha]["min_x"] = min(self.stats[t][alpha]["min_x"], min_x)

    def updateMaxX(self, t, alpha, max_x):
        if self.stats[t][alpha]["max_x"] is None:
            self.stats[t][alpha]["max_x"] = max_x
        else:
            self.stats[t][alpha]["max_x"] = max(self.stats[t][alpha]["max_x"], max_x)

    def updateMinY(self, t, alpha, min_y):
        if self.stats[t][alpha]["min_y"] is None:
            self.stats[t][alpha]["min_y"] = min_y
        else:
            self.stats[t][alpha]["min_y"] = min(self.stats[t][alpha]["min_y"], min_y)

    def updateMaxY(self, t, alpha, max_y):
        if self.stats[t][alpha]["max_y"] is None:
            self.stats[t][alpha]["max_y"] = max_y
        else:
            self.stats[t][alpha]["max_y"] = max(self.stats[t][alpha]["max_y"], max_y)

    def getMinX(self, t, alpha):
        return self.stats[t][alpha]["min_x"]

    def getMaxX(self, t, alpha):
        return self.stats[t][alpha]["max_x"]

    def getMinY(self, t, alpha):
        return self.stats[t][alpha]["min_y"]

    def getMaxY(self, t, alpha):
        return self.stats[t][alpha]["max_y"]

    def print_metrics(self):
        super().print_metrics()
        for i in range(0, len(self.alphas)):
            print(self.metrics_to_string("dense", self.alphas[i]))
        for i in range(0, len(self.alphas)):
            print(self.metrics_to_string("dense_rRT", self.alphas[i]))
        for i in range(0, len(self.alphas)):
            print(self.metrics_to_string("dense_aut", self.alphas[i]))
