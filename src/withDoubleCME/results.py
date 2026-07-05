
def sort_F1_func(e):
    return e["F1"]

class Results(object):
    def __init__(self):
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.seenF1Scores = dict()
        self.seenF1Scores["reg"] = dict()
        self.seenF1Scores["rRT"] = dict()
        self.seenF1Scores["aut"] = dict()
        for i in range(0, 10):
            self.seenF1Scores["reg"][i*10] = dict()
            self.seenF1Scores["rRT"][i*10] = dict()
            self.seenF1Scores["aut"][i*10] = dict()

        self.metrics = dict()
        self.metrics["reg"] = dict()
        self.metrics["rRT"] = dict()
        self.metrics["aut"] = dict()
        for i in range(0, 10):
            self.metrics["reg"][i*10] = dict()
            self.metrics["rRT"][i*10] = dict()
            self.metrics["aut"][i*10] = dict()

        self.all_metrics = dict()
        self.all_metrics["reg"] = dict()
        self.all_metrics["rRT"] = dict()
        self.all_metrics["aut"] = dict()
        for i in range(0, 10):
            self.all_metrics["reg"][i*10] = list()
            self.all_metrics["rRT"][i*10] = list()
            self.all_metrics["aut"][i*10] = list()
            
    def updateMinX(self, min_x):
        if self.min_x is None:
            self.min_x = min_x
        else:
            self.min_x = min(self.min_x, min_x)

    def updateMaxX(self, max_x):
        if self.max_x is None:
            self.max_x = max_x
        else:
            self.max_x = max(self.max_x, max_x)

    def updateMinY(self, min_y):
        if self.min_y is None:
            self.min_y = min_y
        else:
            self.min_y = min(self.min_y, min_y)

    def updateMaxY(self, max_y):
        if self.max_y is None:
            self.max_y = max_y
        else:
            self.max_y = max(self.max_y, max_y)

    def getMinX(self):
        return self.min_x

    def getMaxX(self):
        return self.max_x

    def getMinY(self):
        return self.min_y

    def getMaxY(self):
        return self.max_y

    def getSeenF1Scores(self, t, oversampleAmount):
        return self.seenF1Scores[t][oversampleAmount]

    def updateMetrics(self, t, oversampleAmount, metrics):
        d = self.metrics[t][oversampleAmount]
        for key in metrics.keys():
            d[key] = metrics[key]

    def metrics_to_string(self, t, oversampleAmount):
        m = self.metrics[t][oversampleAmount]
        s = "{},{}".format(t, oversampleAmount)
        for key in ["corrMean", "maeMean", "FPMean", "FNMean", "TPMean", "TNMean", "F1Mean", "HSSMean", "TSSMean"]:
            if key in m:
                s = s + ",{:2f}".format(m[key])
        return s

    def print_metrics(self):
        for i in range(0, 10):
            print(self.metrics_to_string("reg", i*10))
        for i in range(1, 10):
            print(self.metrics_to_string("rRT", i*10))
        for i in range(1, 10):
            print(self.metrics_to_string("aut", i*10))

    def add_all_metric(self, t, oversampleAmount, metrics):
        l = self.all_metrics[t][oversampleAmount]
        m = dict()
        for key in metrics.keys():
            m[key] = metrics[key]
        m["run"] = len(l)
        l.append(m)
        l.sort(key=sort_F1_func)

    def max_F1_info(self):
        key = "F1"
        group = "reg"
        oversample = 0
        max_F1 = self.all_metrics[group][oversample][0][key]
        for all_metric_group in self.all_metrics.keys():
            for all_metric_oversample in self.all_metrics[all_metric_group].keys():
                metric_group = self.all_metrics[all_metric_group][all_metric_oversample]
                if len(metric_group) == 0:
                    continue
                metric_group_max_F1 = metric_group[len(metric_group) - 1][key]
                if metric_group_max_F1 > max_F1:
                    max_F1 = metric_group_max_F1
                    group = all_metric_group
                    oversample = all_metric_oversample
        return (group, oversample, max_F1)

    def median_F1_info(self, group, oversample):
        key = "F1"
        metric_list = self.all_metrics[group][oversample]
        if len(metric_list) % 2 == 1:
            median_idx = int((len(metric_list) - 1) / 2)
        else:
            median_idx = int(len(metric_list) / 2)
        median = metric_list[median_idx]
        return (median["run"], median[key])

    def print_F1(self):
        print("[", end='')
        for all_metric_group in self.all_metrics.keys():
            for all_metric_oversample in self.all_metrics[all_metric_group].keys():
                for run_info in self.all_metrics[all_metric_group][all_metric_oversample]:
                    print(",{};{};{}".format(all_metric_group, all_metric_oversample, run_info["F1"]), end='')
        print("]")