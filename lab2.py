from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import wave
import scipy.signal as signal

(rate, sig) = wav.read("aaa.wav")
mfcc_feat = mfcc(sig, rate)
aaa_c1 = mfcc_feat[:, 1]
aaa_c2 = mfcc_feat[:, 2]
plt.hist(aaa_c1, bins=12, facecolor='pink')
plt.title("histogram współczynnika c1 głoski /a/")
plt.show()
plt.hist(aaa_c2, bins=12, facecolor='pink')
plt.title("histogram współczynnika c2 głoski /a/")
plt.show()

(rate, sig) = wav.read("iii.wav")
mfcc_feat = mfcc(sig, rate)
iii_c1 = mfcc_feat[:, 1]
iii_c2 = mfcc_feat[:, 2]
plt.hist(iii_c1, bins=12, facecolor='green')
plt.title("histogram współczynnika c1 głoski /i/")
plt.show()
plt.hist(iii_c2, bins=12, facecolor='green')
plt.title("histogram współczynnika c2 głoski /i/")
plt.show()

s1 = np.mean(aaa_c1)
s2 = np.mean(aaa_c2)
s3 = np.mean(iii_c1)
s4 = np.mean(iii_c2)
print(s1, s2, s3, s4)

o1 = np.std(aaa_c1)
o2 = np.std(aaa_c2)
o3 = np.std(iii_c1)
o4 = np.std(iii_c2)
print(o1, o2, o3, o4)

r1 = stats.shapiro(aaa_c1)
r2 = stats.shapiro(aaa_c2)
r3 = stats.shapiro(iii_c1)
r4 = stats.shapiro(iii_c2)
print(r1, r2, r3, r4)

plik = wave.open("aaa.wav", 'rb')
bits = plik.readframes(-1)
wav_signal = np.fromstring(bits, "int16")
fp = plik.getframerate()
F, sig = wav.read("aaa.wav")
step = (len(wav_signal)/fp)
length = len(wav_signal)
tm = np.linspace(0, step, num=length)

plt.figure(1)
signal.stft(wav_signal, fp, nperseg=fp * 0.1)
plt.title('Spektrogram głoska /a/')
plt.xlabel('Czas [s]')
plt.ylabel('Częstotliwość [Hz]')
plt.specgram(wav_signal, 1600, fp, noverlap=0)
plt.axis(ymin=0, ymax=500)
plt.show()

wav_fs, signal_wav = wav.read('aaa.wav')
aaa = signal_wav[30000:31024]
aaa = aaa / np.max(np.abs(aaa))
plt.plot(np.arange(30000, 31024), aaa)
plt.xlabel('nr próbki')
plt.ylabel('amplituda')
plt.title('Fragment nagrania dźwięku aaa')
plt.xlim(30000, 30500)
plt.show()

aaa = aaa - np.mean(aaa)
klarnet_autokor = signal.correlate(aaa, aaa)
klarnet_autokor = klarnet_autokor[len(klarnet_autokor)//2:]
klarnet_autokor = klarnet_autokor / klarnet_autokor[0]
plt.plot(klarnet_autokor)
plt.xlabel('Opóźnienie')
plt.ylabel('Autokorelacja')
plt.title('Funkcja autokorelacji głoska /a/')
plt.show()

autokor2 = np.maximum(klarnet_autokor, 0)
bufor = np.zeros(autokor2.shape)
n = len(autokor2) // 2
bufor[::2] = autokor2[:n]
bufor[1::2] = (autokor2[:n] + autokor2[1:n+1]) / 2
autokor2 = np.maximum(autokor2 - bufor, 0)
plt.plot(autokor2)
plt.xlabel('Opóźnienie')
plt.ylabel('Autokorelacja')
plt.title('Zmodyfikowana funkcja autokorelacji głoska /a/')
plt.show()

m = np.argmax(autokor2)
print('Częstotliwość podstawowa aaa = {:.2f} Hz'.format(wav_fs / m))

plik = wave.open("iii.wav", 'rb')
bits = plik.readframes(-1)
wav_signal = np.fromstring(bits, "int16")
fp = plik.getframerate()
F, sig = wav.read("iii.wav")
step = (len(wav_signal)/fp)
length = len(wav_signal)
tm = np.linspace(0, step, num=length)

plt.figure(2)
signal.stft(wav_signal, fp, nperseg=fp * 0.1)
plt.title('Spektrogram głoska /i/')
plt.xlabel('Czas [s]')
plt.ylabel('Częstotliwość [Hz]')
plt.specgram(wav_signal, 1600, fp, noverlap=0)
plt.axis(ymin=0, ymax=500)
plt.show()

wav_fs, signal_wav = wav.read('iii.wav')
iii = signal_wav[30000:31024]
iii = iii / np.max(np.abs(iii))
plt.plot(np.arange(30000, 31024), iii)
plt.xlabel('nr próbki')
plt.ylabel('amplituda')
plt.title('Fragment nagrania dźwięku iii')
plt.xlim(30000, 30500)
plt.show()

iii = iii - np.mean(iii)
klarnet_autokor = signal.correlate(iii, iii)
klarnet_autokor = klarnet_autokor[len(klarnet_autokor)//2:]
klarnet_autokor = klarnet_autokor / klarnet_autokor[0]
plt.plot(klarnet_autokor)
plt.xlabel('Opóźnienie')
plt.ylabel('Autokorelacja')
plt.title('Funkcja autokorelacji głoska /i/')
plt.show()

autokor2 = np.maximum(klarnet_autokor, 0)
bufor = np.zeros(autokor2.shape)
n = len(autokor2) // 2
bufor[::2] = autokor2[:n]
bufor[1::2] = (autokor2[:n] + autokor2[1:n+1]) / 2
autokor2 = np.maximum(autokor2 - bufor, 0)
plt.plot(autokor2)
plt.xlabel('Opóźnienie')
plt.ylabel('Autokorelacja')
plt.title('Zmodyfikowana funkcja autokorelacji głoska /i/')
plt.show()

m = np.argmax(autokor2)
print('Częstotliwość podstawowa iii = {:.2f} Hz'.format(wav_fs / m))