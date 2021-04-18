import wave
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os

def param(list):
   d = {}
   for x in list:
      file = wave.open(x, "rb")
      (rate, sig) = wav.read(x)
      my_mfcc = mfcc(sig, rate)
      my_mfcc = my_mfcc[:, 1]
      key = x
      d[key] = my_mfcc
      file.close()
   return d

file_list = os.listdir(train)
dict = param(file_list)