import numpy as np

try:
    from data_paths import CLASS_FILES
except ImportError:
    from .data_paths import CLASS_FILES

def _load_phase_slice(path, phase):
    data = np.load(path)
    # Copy only one phase, then immediately reduce to PSD features to keep memory low.
    phase_data = np.array(data[:, phase], copy=True)
    if phase_data.ndim >= 3:
        phase_data = np.mean(np.abs(phase_data), axis=2)
    phase_data = phase_data.astype(np.float32, copy=False)
    n_samples = data.shape[0]
    del data
    return phase_data, n_samples

def performance_params(cnf_matrix):
    print(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    print(TN)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    perfs = np.concatenate((TPR[np.newaxis,:],TNR[np.newaxis,:],PPV[np.newaxis,:],NPV[np.newaxis,:],FPR[np.newaxis,:],FNR[np.newaxis,:],FDR[np.newaxis,:],ACC[np.newaxis,:]), axis=1)
    return perfs

class ClassificationTask():
    def __init__(self):
        self.y = None
        self.X = None
        self.classes = None
        self.locations = CLASS_FILES.copy()

    # Possible classification tasks on prereach/reach/grasp
    #   power - precision in general
    #   power - precision in same hand
    #   power - precision no bimanual
    #   classify angles in bimanual
    #   distinction between right and left general
    #   distinction between right and left precision
    #   distinction between right and left power
    #   distinction on angle same hand (only left or only right)
    #   distinction on angle any hand
    #   distintion between phases in precision and power
    #   distinction between unimanual and bimanual tasks

    # Phases are 0 - prereach, 1 - reach, 2 - grasp

    # POWER - PRECISION in general
    def get_task_power_precision(self, phase=1):
        self.X = []
        self.y = []
        self.classes = {0 : "PRECISION", 1: "POWER"}

        for l in self.locations.keys():
            phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
            self.X.append(phase_data)
            if "PRECISION" in l:
                self.y = self.y + [0] * n_samples
            elif "POWER" in l:
                self.y = self.y + [1] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")

    # POWER - PRECISION in same hand
    def get_task_power_precision_hand(self, phase=1, hand="L"):
        self.X = []
        self.y = []
        self.classes = {0 : "PRECISION", 1: "POWER"}

        for l in self.locations.keys():
            if ("_" + hand + "_") in l:
                phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                self.X.append(phase_data)
                if "PRECISION" in l:
                    self.y = self.y + [0] * n_samples
                elif "POWER" in l:
                    self.y = self.y + [1] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    # POWER - PRECISION no bimanual
    def get_task_power_precision_nobi(self, phase=1):
        self.X = []
        self.y = []
        self.classes = {0 : "PRECISION", 1: "POWER"}

        for l in self.locations.keys():
            if "BIMANUAL" not in l:
                phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                self.X.append(phase_data)
                if "PRECISION" in l:
                    self.y = self.y + [0] * n_samples
                elif "POWER" in l:
                    self.y = self.y + [1] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    # classify angles in bimanual
    def get_task_angles_bimanual(self, phase=1):
        self.X = []
        self.y = []
        self.classes = {0 : "45", 1: "135", 2: "45_135", 3: "135_45"}

        for l in self.locations.keys():
            if "BIMANUAL" in l:
                phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                self.X.append(phase_data)
                if "135_45" in l:
                    self.y = self.y + [3] * n_samples
                elif "45_135" in l:
                    self.y = self.y + [2] * n_samples
                elif "135" in l:
                    self.y = self.y + [1] * n_samples
                elif "45" in l:
                    self.y = self.y + [0] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    # distinction between right and left general
    def get_task_left_right(self, phase=1):
        self.X = []
        self.y = []
        self.classes = {0 : "L", 1: "R"}

        for l in self.locations.keys():
            if "UNIMANUAL" in l:
                phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                self.X.append(phase_data)
                if "_L_" in l:
                    self.y = self.y + [0] * n_samples
                elif "_R_" in l:
                    self.y = self.y + [1] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    # distinction between right and left precision
    def get_task_left_right_precision(self, phase=1):
        self.X = []
        self.y = []
        self.classes = {0 : "L", 1: "R"}

        for l in self.locations.keys():
            if "PRECISION_UNIMANUAL" in l:
                phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                self.X.append(phase_data)
                if "_L_" in l:
                    self.y = self.y + [0] * n_samples
                elif "_R_" in l:
                    self.y = self.y + [1] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    # distinction between right and left precision
    def get_task_left_right_power(self, phase=1):
        self.X = []
        self.y = []
        self.classes = {0 : "L", 1: "R"}

        for l in self.locations.keys():
            if "POWER_UNIMANUAL" in l:
                phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                self.X.append(phase_data)
                if "_L_" in l:
                    self.y = self.y + [0] * n_samples
                elif "_R_" in l:
                    self.y = self.y + [1] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")

    # distinction on angle same hand (only left or only right)
    def get_task_angles_hand(self, phase=1, hand="L", pp="ALL"):
        if pp == "ALL": 
            pp = "_"
        self.X = []
        self.y = []
        self.classes = {0 : "0", 1: "45", 2: "90", 3: "135"}

        for l in self.locations.keys():
            if pp in l:
                if ("_"+ hand + "_") in l:
                    phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                    self.X.append(phase_data)
                    if "_0" in l:
                        self.y = self.y + [0] * n_samples
                    elif "_45" in l:
                        self.y = self.y + [1] * n_samples
                    elif "_90" in l:
                        self.y = self.y + [2] * n_samples
                    elif "_135" in l:
                        self.y = self.y + [3] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    # distinction on angle any hand
    def get_task_angles_any_hand(self, phase=1, pp="ALL"):
        if pp == "ALL": 
            pp = "_"
        self.X = []
        self.y = []
        self.classes = {0 : "0", 1: "45", 2: "90", 3: "135"}

        for l in self.locations.keys():
            if pp in l:
                if "UNIMANUAL" in l:
                    phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                    self.X.append(phase_data)
                    if "_0" in l:
                        self.y = self.y + [0] * n_samples
                    elif "_45" in l:
                        self.y = self.y + [1] * n_samples
                    elif "_90" in l:
                        self.y = self.y + [2] * n_samples
                    elif "_135" in l:
                        self.y = self.y + [3] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    # distintion between phases in precision and power
    def get_task_phases(self, pp="ALL"):
        if pp == "ALL": 
            pp = "_"
        self.X = []
        self.y = []
        self.classes = {0 : "PREREACH", 1: "REACH", 2: "GRASP"}

        for l in self.locations.keys():
            if pp in l:
                data = np.load(self.locations[l])
                n_samples = data.shape[0]

                phase0 = np.array(data[:, 0], copy=True)
                if phase0.ndim >= 3:
                    phase0 = np.mean(np.abs(phase0), axis=2)
                self.X.append(phase0.astype(np.float32, copy=False))
                self.y = self.y + [0] * n_samples
                phase1 = np.array(data[:, 1], copy=True)
                if phase1.ndim >= 3:
                    phase1 = np.mean(np.abs(phase1), axis=2)
                self.X.append(phase1.astype(np.float32, copy=False))
                self.y = self.y + [1] * n_samples
                phase2 = np.array(data[:, 2], copy=True)
                if phase2.ndim >= 3:
                    phase2 = np.mean(np.abs(phase2), axis=2)
                self.X.append(phase2.astype(np.float32, copy=False))
                self.y = self.y + [2] * n_samples
                del data

        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    # distintion between unimanual and bimanual tasks
    def get_task_unimanual_bimanual(self, phase=1):
        self.X = []
        self.y = []
        self.classes = {0 : "UNIMANUAL", 1: "BIMANUAL"}

        for l in self.locations.keys():
            if "PRECISION" in l:
                phase_data, n_samples = _load_phase_slice(self.locations[l], phase)
                self.X.append(phase_data)
                if "UNIMANUAL" in l:
                    self.y = self.y + [0] * n_samples
                elif "BIMANUAL" in l:
                    self.y = self.y + [1] * n_samples
        self.X = np.concatenate(self.X)
        self.y = np.array(self.y)
        print(self.X.shape[0], "samples loaded")
    
    def get_task_multioutput_test(self, phase=1):
        self.X = []
        self.y = []
        self.classes = {0 : "UNIMANUAL", 1: "BIMANUAL", 
                        2: "LEFT", 3: "RIGHT",
                        4: "L0", 5: "L45", 6: "L90", 7: "L135", 
                        8: "R0", 9: "R45", 10: "R90", 11: "R135", 
                        12: "POWER", 13: "PRECISION"
                        }

        for l in self.locations.keys():
            data = np.load(self.locations[l])
            self.X.append(data[:,phase])
            one_hots = np.zeros((data.shape[0], 14))

            if "UNIMANUAL" in l:
                one_hots[:,0] = 1
                if "_L_" in l:
                    one_hots[:,2] = 1
                    if "_0" in l:
                        one_hots[:,4] = 1
                    elif "_45" in l:
                        one_hots[:,5] = 1
                    elif "_90" in l:
                        one_hots[:,6] = 1
                    elif "_135" in l:
                        one_hots[:,7] = 1
                elif "_R_" in l: 
                    one_hots[:,3] = 1
                    if "_0" in l:
                        one_hots[:,8] = 1
                    elif "_45" in l:
                        one_hots[:,9] = 1
                    elif "_90" in l:
                        one_hots[:,10] = 1
                    elif "_135" in l:
                        one_hots[:,11] = 1
            elif "BIMANUAL" in l: 
                one_hots[:,1] = 1
                if "135_45" in l:
                    one_hots[:,7] = 1
                    one_hots[:,9] = 1
                elif "45_135" in l:
                    one_hots[:,5] = 1
                    one_hots[:,11] = 1
                elif "135" in l:
                    one_hots[:,7] = 1
                    one_hots[:,11] = 1
                elif "45" in l:
                    one_hots[:,5] = 1
                    one_hots[:,9] = 1
            
            if "POWER" in l:
                one_hots[:,12] = 1
            elif "PRECISION" in l:
                one_hots[:,13] = 1
            
            self.y.append(one_hots)
        self.X = np.concatenate(self.X)
        self.y = np.concatenate(self.y)
        print(self.X.shape, "samples loaded")
        print(self.y.shape, "output shape")

    def get_task_multioutput(self, phase=2):
        self.X = []
        self.y = []
        self.classes = {0 : "UNIMANUAL", 1: "BIMANUAL", 
                        2: "LEFT", 3: "RIGHT",
                        4: "L0", 5: "L45", 6: "L90", 7: "L135", 
                        8: "R0", 9: "R45", 10: "R90", 11: "R135", 
                        12: "PRECISION", 13: "POWER"
                        }

        for l in self.locations.keys():
            data = np.load(self.locations[l])
            self.X.append(data[:,phase])
            one_hots = np.zeros((data.shape[0], 14))

            if "UNIMANUAL" in l:
                one_hots[:,0] = 1
                if "_L_" in l:
                    one_hots[:,2] = 1
                    if "_0" in l:
                        one_hots[:,4] = 1
                    elif "_45" in l:
                        one_hots[:,5] = 1
                    elif "_90" in l:
                        one_hots[:,6] = 1
                    elif "_135" in l:
                        one_hots[:,7] = 1
                elif "_R_" in l: 
                    one_hots[:,3] = 1
                    if "_0" in l:
                        one_hots[:,8] = 1
                    elif "_45" in l:
                        one_hots[:,9] = 1
                    elif "_90" in l:
                        one_hots[:,10] = 1
                    elif "_135" in l:
                        one_hots[:,11] = 1
            elif "BIMANUAL" in l: 
                one_hots[:,1] = 1
                if "135_45" in l:
                    one_hots[:,7] = 1
                    one_hots[:,9] = 1
                elif "45_135" in l:
                    one_hots[:,5] = 1
                    one_hots[:,11] = 1
                elif "135" in l:
                    one_hots[:,7] = 1
                    one_hots[:,11] = 1
                elif "45" in l:
                    one_hots[:,5] = 1
                    one_hots[:,9] = 1
            
            if "PRECISION" in l:
                one_hots[:,12] = 1
            elif "POWER" in l:
                one_hots[:,13] = 1
            
            self.y.append(one_hots)
        self.X = np.concatenate(self.X)
        self.y = np.concatenate(self.y)
        print(self.X.shape, "samples loaded")
        print(self.y.shape, "output shape")


if __name__ == "__main__":
    test = ClassificationTask()
    test.get_task_multioutput()


if __name__ == "__main__a":
    test = ClassificationTask()

    # 4975 samples
    test.get_task_power_precision()

    # 2087 samples
    test.get_task_power_precision_hand(hand="L")
    # 2098 samples
    test.get_task_power_precision_hand(hand="R")

    # 4185 samples
    test.get_task_power_precision_nobi()

    # 790 samples
    test.get_task_angles_bimanual()

    # 4185 samples
    test.get_task_left_right()

    # 1702 samples
    test.get_task_left_right_precision()

    # 2483 samples
    test.get_task_left_right_power()

    # different tasks for angles and hands
    test.get_task_angles_hand(hand="L", pp="ALL")       # 2087 samples
    test.get_task_angles_hand(hand="L", pp="PRECISION") # 854 samples
    test.get_task_angles_hand(hand="L", pp="POWER")     # 1233 samples
    test.get_task_angles_hand(hand="R", pp="ALL")       # 2098 samples
    test.get_task_angles_hand(hand="R", pp="PRECISION") # 848 samples
    test.get_task_angles_hand(hand="R", pp="POWER")     # 1250 samples
    
    # different tasks for angles
    test.get_task_angles_any_hand(pp="ALL")       # 4185 samples
    test.get_task_angles_any_hand(pp="PRECISION") # 1702 samples
    test.get_task_angles_any_hand(pp="POWER")     # 2483 samples

    # different tasks for phases
    test.get_task_phases(pp="ALL")         # 14925 samples
    test.get_task_phases(pp="PRECISION")   # 7476 samples
    test.get_task_phases(pp="POWER")       # 7449 samples

    # 2492 samples
    test.get_task_unimanual_bimanual()
