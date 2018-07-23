import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

#software data
#aflsingle = pd.read_csv('AFLSingle.csv', delimiter=' ', header=None)
kleesingle = pd.read_csv('KLEEsingle.csv', delimiter=' ', header=None)
#y_afl = aflsingle.iloc[:, 2].values
#y_afl = (y_afl >= 1)
y_klee = kleesingle.iloc[:, 2].values
y_klee = (y_klee >= 1)

'''
# modifiying afl and klee for diff
commands_diff = ['CAT', 'ECHO', 'EXPAND', 'EXPR', 'FMT', 'ID', 
            'KILL', 'PATHCHK', 'PINKY', 'PRINTF', 'PTX', 'SEQ', 'STAT',
            'TAIL', 'TEST', 'TOUCH', 'UNEXPAND', 'UNIQ', 'WHO']
data_dict_dif = []
for cd in aflsingle[0]:
    for i in range(0,20):
        if commands_diff[i].lower()== cd :
            data_dict_dif.append(temp_mean)
'''

#input data preprocessing
class tosent:
    def __init__ (self,command,version):
        self.command = command
        self.version = version
    def get_data(self):
        dataset = pd.read_csv('tosent/' + self.command + '/' + self.version + '/AvailableVariablesFiltered.txt', header=None)
        dataset= dataset.iloc[:, 3:-1]
        dataset.columns = ["trace_length", "global_nesting_depth", "call_depth", "no_of_local_variable", "taint_count"]
        return dataset
    def get_train_data(self):
        dataset_final= self.get_data().iloc[:, :].values
        return dataset_final

'''
commands_same = ['CHMOD', 'CUT', 'HEAD', 'DF', 'DU', 'LN', 'LS', 'NL', 'PR', 'SUM', 'WC']

commands_diff = ['CAT', 'ECHO', 'EXPAND', 'EXPR', 'FMT', 'ID', 'KILL', 'PATHCHK', 'PINKY', 'PRINTF',
                 'PTX', 'SEQ', 'STAT', 'TAIL', 'TEST', 'TOUCH', 'UNEXPAND', 'UNIQ', 'WHO']

data_dict_same= []
data_dict_diff= []
'''

commands = ['CAT', 'CHMOD', 'CUT', 'DF', 'DU', 'ECHO', 'EXPAND', 'EXPR', 'FMT', 'HEAD', 'ID', 
            'KILL', 'LN', 'LS', 'NL', 'PATHCHK', 'PINKY', 'PR', 'PRINTF', 'PTX', 'SEQ', 'STAT',
            'SUM', 'TAIL', 'TEST', 'TOUCH', 'UNEXPAND', 'UNIQ', 'WC', 'WHO']

data_dict= []

for name in commands:
    for i in range(1, 5):
        scan_data= tosent(name, str(i))
        temp=scan_data.get_train_data()
        temp_mean=temp.mean(0)
        temp_median = np.median(temp, axis= 0)
        temp_max= temp.max(0)
        temp_min= temp.min(0)
        final_temp= []
        final_temp.append(temp_mean[0])
        final_temp.append(temp_max[1])
        final_temp.append(temp_mean[2])
        final_temp.append(temp_max[3])
        final_temp.append(temp_max[4])
        data_dict.append(final_temp)
        
for name in commands_diff:
    for i in range(1, 5):
        scan_data= tosent(name, str(i))
        temp=scan_data.get_train_data()
        temp_mean=temp.mean(0)
        temp_median = np.median(temp, axis= 0)
        temp_max= temp.max(0)
        temp_min= temp.min(0)
        final_temp= []
        final_temp.append(temp_mean[0])
        final_temp.append(temp_max[1])
        final_temp.append(temp_mean[2])
        final_temp.append(temp_max[3])
        final_temp.append(temp_max[4])
        data_dict_diff.append(final_temp)
        
#from apyori import apriori
#rules = apriori(data_dict, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)






