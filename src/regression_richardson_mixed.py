from regression import *

class RegressionRichardsonMixed(Regression):
    def __init__(self):
        self.lossWeightAlpha = 0.804158527

    def getFirstStageInput(self, csvFile):
        data = readCSVFile(csvFile)
    
        numElements = len(data)
        inputData_x = np.empty((numElements, 18), dtype=np.float64)
        inputData_y = np.empty((numElements, 1), dtype=np.float64)
    
        for i in range(0, numElements):
            elem = data[i]
        
            inputData_x[i][0] = float(elem["donki_speed"])
            inputData_x[i][1] = float(elem["donki_ha"])
            inputData_x[i][2] = float(elem["longitude"])
            inputData_x[i][3] = float(elem["latitude"])
            inputData_x[i][4] = float(elem["Accel"])
            inputData_x[i][5] = float(elem["Type_2_Area"])
            inputData_x[i][6] = float(elem["2nd_order_speed_final"])
            inputData_x[i][7] = float(elem["2nd_order_speed_20R"])
            inputData_x[i][8] = float(elem["Central_PA"])
            inputData_x[i][9] = float(elem["MPA"])
            inputData_x[i][10] = float(elem["sunspots"])
            inputData_x[i][11] = float(elem["CMEs_past_month"])
            inputData_x[i][12] = float(elem["CMEs_past_9_hours"])
            inputData_x[i][13] = float(elem["CMEs_over_1000_past_9_hrs"])
            inputData_x[i][14] = float(elem["Max_speed_past_day"])
            inputData_x[i][15] = float(elem["V log V"])
            inputData_x[i][16] = float(elem["halo"])
            inputData_x[i][17] = float(elem["diffusive_shock"])
        
            inputData_y[i][0] = float(elem["100MeV_peak_intensity_ln"])
    
        return (inputData_x, inputData_y)
