import pyaudio
import wave

wave_file = wave.open("speach.wav", 'rb')
p = pyaudio.PyAudio()
f = p.get_format_from_width(wave_file.getsampwidth())
c = wave_file.getnchannels()
r = wave_file.getframerate()
stream = p.open(format=f, channels=c, rate=r, output=True)
data = wave_file.readframes(1024)

while data != '':
   stream.write(data)
   data = wave_file.readframes(1024)

stream.stop_stream()
stream.close()
p.terminate()

import wave
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
plik = wave.open("speach.wav", 'rb')
bits = plik.readframes(-1)
wav_signal = np.fromstring(bits, "int16")
fp = plik.getframerate()
F, sig = scipy.io.wavfile.read("speach.wav")
step = (len(wav_signal)/fp)
length = len(wav_signal)
tm = np.linspace(0, step, num=length)
plt.figure(1)
plt.title('Oscillograph')
plt.plot(tm, wav_signal)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()

import wave
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy.signal as signal
plik = wave.open("speech.wav", 'rb')
bits = plik.readframes(-1)
wav_signal = np.fromstring(bits, "int16")
fp = plik.getframerate()
F, sig = scipy.io.wavfile.read("speech.wav")
step = (len(wav_signal)/fp)
length = len(wav_signal)
tm = np.linspace(0, step, num=length)

plt.figure(1)
signal.stft(wav_signal, fp, nperseg=fp * 0.2)
plt.title('Spectrogram window_time: 200ms')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.specgram(wav_signal, 8820, fp, noverlap=0)
plt.axis(ymin=0, ymax=8000)
plt.show()

plt.figure(2)
signal.stft(wav_signal, fp, nperseg=fp * 0.1)
plt.title('Spectrogram window_time: 100ms')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.specgram(wav_signal,  4410, fp, noverlap=0)
plt.axis(ymin=0, ymax=8000)
plt.show()

plt.figure(3)
signal.stft(wav_signal, fp, nperseg=fp * 0.05)
plt.title('Spectrogram window_time: 50ms')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.specgram(wav_signal, 2205, fp, noverlap=0)
plt.axis(ymin=0, ymax=8000)
plt.show()

plt.figure(4)
signal.stft(wav_signal, fp, nperseg=fp * 0.02)
plt.title('Spectrogram window_time: 20ms')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.specgram(wav_signal,  882, fp, noverlap=0)
plt.axis(ymin=0, ymax=8000)
plt.show()

plt.figure(5)
signal.stft(wav_signal, fp, nperseg=fp * 0.01)
plt.title('Spectrogram window_time: 10ms')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.specgram(wav_signal, 441, fp, noverlap=0)
plt.axis(ymin=0, ymax=8000)
plt.show()

plt.figure(6)
signal.stft(wav_signal, fp, nperseg=fp * 0.005)
plt.title('Spectrogram window_time: 5ms')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.specgram(wav_signal, 220, fp, noverlap=0)
plt.axis(ymin=0, ymax=8000)
plt.show()