import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

#software data
aflsingle = pd.read_csv('AFLSingle.csv', delimiter=' ', header=None)
kleesingle = pd.read_csv('KLEEsingle.csv', delimiter=' ', header=None)
y_afl = aflsingle.iloc[:, 2].values
y_afl = (y_afl >= 1)
y_klee = kleesingle.iloc[:, 2].values
y_klee = (y_klee >= 1)


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
        
        

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_dict, y_klee, test_size = 0.175)

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
sc = StandardScaler()
sc = sc.fit(X_train)
####joblib.dump(sc, "AFL_models/sc85.save") 

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
classifier.add(Dense(output_dim = 1000, init = 'uniform', activation = 'relu', input_dim = 5))
classifier.add(Dense(output_dim = 1000, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

##classifier.save('KLEE_models/model80-85_new.h5')

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

per = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])