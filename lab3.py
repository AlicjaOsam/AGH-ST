from python_speech_features import mfcc
import scipy.io.wavfile as wav
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math

def analyze(MFCC, n, n_iter):
    GMM = GaussianMixture(init_params='random', n_components=n, max_iter=n_iter, covariance_type='diag')
    GMM.fit(MFCC)
    return GMM

def probability(MFCC, GMM, m):
    c = MFCC[:, m]
    y, x, z = plt.hist(c, bins='auto')
    d = len(GMM.means_[:, m])
    n = []
    sg = []
    for i in range(d):
        n.append(GMM.means_[i, m]*GMM.weights_[i])
        sg.append(np.sqrt(GMM.covariances_[i, m]*GMM.weights_[i]))
    ni = sum(n)
    sigma = sum(sg)
    x = np.arange(ni-3*sigma, ni+3*sigma, 0.001)
    gauss = 100*stats.norm.pdf(x, ni, sigma)
    y_max = y.max()
    gauss = gauss*y_max/np.max(gauss)
    plt.plot(x, gauss)
    plt.show()
    BIC = GMM.bic(MFCC)
    print(BIC)


(rate, sig) = wav.read("aaa.wav")
MFCC_A = mfcc(sig, rate)

(rate, sig) = wav.read("iii.wav")
MFCC_I = mfcc(sig, rate)

w = []
k = range(1, 100)
for i in range(1, 100):
    GMM_A = analyze(MFCC_A, 6, i)
    AIC = GMM_A.aic(MFCC_A)
    w.append(AIC)
plt.plot(k, w)
plt.xlabel('Liczba iteracji')
plt.ylabel('Kryterium informacyjne: aic')
plt.title('AIC')
plt.show()

w = []
j = [1, 2, 3, 4, 5, 6, 7, 8]
for t in j:
    GMM_A = analyze(MFCC_A, t, 20)
    BIC = GMM_A.bic(MFCC_A)
    w.append(BIC)
plt.plot(j, w)
plt.xlabel('Liczba komponentów')
plt.ylabel('Kryterium informacyjne: bic')
plt.title('BIC')
plt.show()

GMM_A = analyze(MFCC_A, 1, 20)
probability(MFCC_A, GMM_A, 1)
GMM_A = analyze(MFCC_A, 2, 20)
probability(MFCC_A, GMM_A, 1)
GMM_A = analyze(MFCC_A, 4, 20)
probability(MFCC_A, GMM_A, 1)
GMM_A = analyze(MFCC_A, 8, 20)
probability(MFCC_A, GMM_A, 1)

GMM_A = analyze(MFCC_A, 8, 5)
probability(MFCC_A, GMM_A, 1)
GMM_A = analyze(MFCC_A, 8, 10)
probability(MFCC_A, GMM_A, 1)
GMM_A = analyze(MFCC_A, 8, 20)
probability(MFCC_A, GMM_A, 1)
GMM_A = analyze(MFCC_A, 8, 50)
probability(MFCC_A, GMM_A, 1)
GMM_A = analyze(MFCC_A, 8, 100)
probability(MFCC_A, GMM_A, 1)


GMM_A = analyze(MFCC_A, 6, 20)
probability(MFCC_A, GMM_A, 1)
GMM_I = analyze(MFCC_I, 6, 20)
probability(MFCC_I, GMM_I, 1)
p1 = GMM_A.score(MFCC_A)
print(p1)
p2 = GMM_A.score(MFCC_I)
print(p2)
p3 = GMM_I.score(MFCC_I)
print(p3)
p4 = GMM_I.score(MFCC_A)
print(p4)

(rate, sig) = wav.read("a1.wav")
MFCC_A1 = mfcc(sig, rate)
(rate, sig) = wav.read("i1.wav")
MFCC_I1 = mfcc(sig, rate)
probability(MFCC_A1, GMM_A, 1)
probability(MFCC_I1, GMM_I, 1)
p1 = GMM_A.score(MFCC_A1)
p2 = GMM_A.score(MFCC_I1)
p3 = GMM_I.score(MFCC_I1)
p4 = GMM_I.score(MFCC_A1)
p1 = math.exp(p1)
p2 = math.exp(p2)
p3 = math.exp(p3)
p4 = math.exp(p4)
lam = 1/2
r1 = p1*lam/(p1*lam+p4*lam) #GMM_A/MFCC_A
r2 = p3/(p2+p3)   #GMM_I/MFCC_I
r3 = p4/(p1+p4) #GMM_I/MFCC_A
r4 = p2/(p2+p3) #GMM_A/MFCC_I
print(r1)
print(r2)
print(r3)
print(r4)