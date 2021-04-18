import wave
from sklearn.mixture import GaussianMixture
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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

def gmm_model(matrix, n, n_iter, cov_type, ran_state):
    model = GaussianMixture(n, max_iter=n_iter, covariance_type=cov_type, random_state=ran_state, n_init=5,
                            tol=1e-9).fit(matrix)
    return model

def cross_test(data, n, n_iter, cov_type, ran_state):
    kf = KFold(n_splits=5)
    my_accuracy = []
    for train_index, test_index in kf.split(data):
        numbers1 = []
        data_train = []
        data_test = []
        test_number = []
        train_number = []
        model_matrix = []
        numbers = []
        second = []
        for r in test_index:
            data_test.append(data[r])
            r = r - test_index[0]
            for nr in range(0, 10):
                if r <= len(data_test):
                    test_number.append(mfc_dictionary[str(data_test[r]) + str(nr)])
        i = 0
        for m in train_index:
            data_train.append(data[m])
            for nr in range(0, 10):
                if m <= (len(data_train) + len(data_test)):
                    train_number.append(mfc_dictionary[str(data_train[i]) + str(nr)])
            i += 1

        for h in train_number:
            numbers.append(h[0])

        for h in test_number:
            numbers1.append(h[0])

        dl = []
        for x in range(0, 10):
            first = []
            for nr in range(x, len(numbers), 10):
                first.append(numbers[nr])
            # print(len(first))
            dl.append(len(first))
            second.append(first)

        for nr in range(0, 10):
            xx = []
            for a in range(0, dl[nr]):
                my_train = (second[nr][a])
                xx.append(my_train)
            good = np.concatenate(xx)
            model = gmm_model(good, n, n_iter, cov_type, ran_state)
            model_matrix.append(model)

        y_true = []
        y_pred = []
        for x in numbers1:
            score_matrix = []
            for y in range(0, 10):
                prediction = model_matrix[y].score(x)
                score_matrix.append(prediction)
            for z in range(0, 10):
                if score_matrix[z] == max(score_matrix):
                    y_pred.append(z)

        for x in range(0, len(test_index)):
            for y in range(0, 10):
                y_true.append(y)
        print(y_pred)
        print(y_true)

        my_accuracy.append(accuracy_score(y_true, y_pred) * 100)
        print(my_accuracy)
        accuracy = np.mean(my_accuracy)
    return accuracy

def eval_data(data):
  mfc = []
  d = {}
  i = 0
  for x in data:
      file = wave.open(x, "rb")
      (rate, sig) = wav.read(x)
      my_mfc = mfcc(sig, rate)
      mfc.append(my_mfc)
      d[x[68:75]] = my_mfc
      file.close()
  return d, mfc

def full_training(data, n, n_iter, cov_type, ran_state):
    data_train = []
    train_number = []
    train_number = []
    numbers = []
    second = []
    model_matrix = []
    for x in range(0, 22):
        data_train.append(data[x])
        for nr in range(0, 10):
            train_number.append(mfc_dictionary[str(data_train[x]) + str(nr)])
    for h in train_number:
        numbers.append(h[0])
    dl = []
    for x in range(0, 10):
        first = []
        for nr in range(x, len(numbers), 10):
            first.append(numbers[nr])
        dl.append(len(first))
        second.append(first)
    for nr in range(0, 10):
        xx = []
        for a in range(0, dl[nr]):
            my_train = (second[nr][a])
            xx.append(my_train)
        good = np.concatenate(xx)
        model = gmm_model(good, n, n_iter, cov_type, ran_state)
        model_matrix.append(model)
    return model_matrix

def test(model_matrix, eval_matrix):
    y_true = []
    y_pred = []
    max_score = []
    for x in eval_matrix:
        score_matrix = []
        for y in range(0, 10):
            prediction = model_matrix[y].score(x)
            score_matrix.append(prediction)
        max_score.append(max(score_matrix))
        for z in range(0, 10):
            if score_matrix[z] == max(score_matrix):
                y_pred.append(z)
    for x in range(0, ):
        for y in range(0, 10):
            y_true.append(y)
    return y_pred, max_score

file_list = os.listdir(train)
mfc_dictionary, matrix_list = create_data(file_list)

accu = cross_test(get_speaker(), 8, 1, 'diag', 2)
print(accu)

eval_path='C:/eval'
eval_list = [os.path.join(eval_path,f) for f in os.listdir(eval_path) if f.endswith('.wav')]
eval_dict, eval_mx = eval_data(eval_list)
models = full_training(get_speaker(), 7, 3, 'diag', 34)
predictions, scores = test(models, eval_mx)
print(predictions)
print(scores)