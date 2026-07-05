from regression import *

class RegressionDenseLoss(Regression):
    def __init__(self):
        super().__init__()
        self.denseLossAlpha = 0.9

