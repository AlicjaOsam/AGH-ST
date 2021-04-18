import wave
from sklearn.mixture import GaussianMixture
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import math

def create_data(data):
   mfc = []
   d = {}
   i = 0
   w = 0
   for x in data:
       file = wave.open(x, "rb")
       (rate, sig) = wav.read(x)
       my_mfc = mfcc(sig, rate)
       mfc.append(my_mfc)
       key = x
       d[key[0:3] + str(w)] = (my_mfc, i)
       i += 1
       w += 1
       if w <= 9:
           w = w
       else:
           w = 0
       file.close()
   return d, mfc

def get_speaker():
   temp = []
   temp2 = []
   all_keys = mfc_dictionary.keys()
   for g in all_keys:
       temp.append(g[0:3])
   for t in range(0, 220, 10):
       temp2.append(temp[t])
   return temp2

def gmm_model(matrix, n, n_iter, cov_type):
   model = GaussianMixture(n, max_iter=n_iter, covariance_type=cov_type, init_params='random', n_init=5, tol=1e-9).fit(
       matrix)
   return model

def cross_test(data, n, n_iter, cov_type):
   kf = KFold(n_splits=5)
   data_train = []
   data_test = []
   test_number = []
   train_number = []
   model_matrix = []
   my_accuracy = []
   k = 0
   i = 0
   for nr in range(0, 10):
       a = 0
       for train_index, test_index in kf.split(data):
           # print("Train indexes: ", train_index, "Test indexes: ", test_index)
           for r in test_index:
               data_test.append(data[r])
               r = r - test_index[0]
               test_number.append(mfc_dictionary[str(data_test[r]) + str(nr)])
           for m in train_index:
               data_train.append(data[m])
               train_number.append(mfc_dictionary[str(data_train[len(data_train) - 1]) + str(nr)])
               i += 1
           i = 0
           numbers = []
           for h in train_number:
               numbers.append(h[0])
           my_train = np.concatenate(numbers, axis=0)
       model = gmm_model(my_train, n, n_iter, cov_type)
       model_matrix.append(model)
   t = 0
   conf_matrix = [[0] * 10 for i in range(10)]
   for i in range(0, 10):
       for j in range(0, 22):
           score_matrix = []
           a = data[j]
           b = mfc_dictionary[str(data[j]) + str(i)]
           c = b[0]
           for k in range(0, 10):
               prediction = model_matrix[k].score(c)
               score_matrix.append(prediction)
           for x in range(0, 10):
               if score_matrix[x] == max(score_matrix):
                   conf_matrix[i][x] = conf_matrix[i][x] + 1
                   if x == i:
                       t = t + 1
   for x in range(0, 10):
       for y in range(0, 10):
           a = conf_matrix[x][x]
           b = 22 - a
           y_true = []
           y_pred = []
           for z in range(0, a):
               y_pred.append(x)
               y_true.append(x)
           for z in range(0, b):
               y_pred.append(-1)
               y_true.append(x)
           print("p",y_pred)
           print("t",y_true)
           my_accuracy.append(accuracy_score(y_true, y_pred))
   my_accuracy = np.mean(my_accuracy)
   return my_accuracy*100

file_list = os.listdir(train)
mfc_dictionary, matrix_list = create_data(file_list)

accu = cross_test(get_speaker(), 8, 1, 'diag')
print(accu)