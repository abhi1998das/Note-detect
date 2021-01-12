from pydub import AudioSegment
import sys
import random
import math
import os     #
import pyaudio
from scipy import signal
from socket import *
from random import *
import numpy
from scipy.signal import blackmanharris, fftconvolve
from numpy import argmax, sqrt, mean, diff, log
import numpy as np
from numpy.fft import rfft
from numpy import asarray, argmax, mean, diff, log, copy
from scipy.signal import correlate, kaiser, decimate


def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res


def build_default_tuner_range():

    return {65.41: 'C2',
            69.30: 'C2#',
            73.42: 'D2',
            77.78: 'E2b',
            82.41: 'E2',
            87.31: 'F2',
            92.50: 'F2#',
            98.00: 'G2',
            103.80: 'G2#',
            110.00: 'A2',
            116.50: 'B2b',
            123.50: 'B2',
            130.80: 'C3',
            138.60: 'C3#',
            146.80: 'D3',
            155.60: 'E3b',
            164.80: 'E3',
            174.60: 'F3',
            185.00: 'F3#',
            196.00: 'G3',
            207.70: 'G3#',
            220.00: 'A3',
            233.10: 'B3b',
            246.90: 'B3',
            261.60: 'C4',
            277.20: 'C4#',
            293.70: 'D4',
            311.10: 'E4b',
            329.60: 'E4',
            349.20: 'F4',
            370.00: 'F4#',
            392.00: 'G4',
            415.30: 'G4#',
            440.00: 'A4',
            466.20: 'B4b',
            493.90: 'B4',
            523.30: 'C5',
            554.40: 'C5#',
            587.30: 'D5',
            622.30: 'E5b',
            659.30: 'E5',
            698.50: 'F5',
            740.00: 'F5#',
            784.00: 'G5',
            830.60: 'G5#',
            880.00: 'A5',
            932.30: 'B5b',
            987.80: 'B5',
            1047.00: 'C6',
            1109.0: 'C6#',
            1175.0: 'D6',
            1245.0: 'E6b',
            1319.0: 'E6',
            1397.0: 'F6',
            1480.0: 'F6#',
            1568.0: 'G6',
            1661.0: 'G6#',
            1760.0: 'A6',
            1865.0: 'B6b',
            1976.0: 'B6',
            2093.0: 'C7'
            }


def loudness(chunk):
    data = numpy.array(chunk, dtype=float) / 32768.0
    ms = math.sqrt(numpy.sum(data ** 2.0) / len(data))
    if ms < 10e-8:
        ms = 10e-8
    return 10.0 * math.log(ms, 10.0)


def freq_from_autocorr(raw_data_signal, fs):
    corr = fftconvolve(raw_data_signal, raw_data_signal[::-1], mode='full')

    corr = corr[int(len(corr)/2):]
    d = diff(corr)
    if len(find(d > 0)) == 0:
        return 0

    start = find(d > 0)[0]

    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    # print(px)
    return fs / px


def freq_from_fft(signal, fs):
    """
    Estimate frequency from peak of FFT
    Pros: Accurate, usually even more so than zero crossing counter
    (1000.000004 Hz for 1000 Hz, for instance).  Due to parabolic
    interpolation being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with signal length
    Cons: Doesn't find the right value if harmonics are stronger than
    fundamental, which is common.
    """
    signal = asarray(signal)

    N = len(signal)

    # Compute Fourier transform of windowed signal
    windowed = signal * kaiser(N, 100)
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i_peak = argmax(abs(f))  # Just use this value for less-accurate result
    i_interp = parabolic(log(abs(f)), i_peak)[0]

    # Convert to equivalent frequency
    return fs * i_interp / N  # Hz


def find_nearest(array, value):
    index = (numpy.abs(array - value)).argmin()
    return array[index]


def closest_value_index(array, guessValue):
    # Find closest element in the array, value wise
    closestValue = find_nearest(array, guessValue)
    # Find indices of closestValue
    indexArray = numpy.where(array == closestValue)
    # Numpys 'where' returns a 2D array with the element index as the value
    return indexArray[0][0]


sound = AudioSegment.from_file("wavfiles/c6.wav")

tunerNotes = build_default_tuner_range()
# print(tunerNotes)
#samples = sound.get_array_of_samples()
frequencies = numpy.array(sorted(tunerNotes.keys()))
# print(frequencies)
soundgate = 19
slices = sound[::500]


for sl in slices:
    raw_data_signal = sl.get_array_of_samples()
    signal_level = round(abs(loudness(raw_data_signal)), 2)
    inputnote = 0

    try:
        # find the freq from the audio sample
        inputnote = round(freq_from_autocorr(
            raw_data_signal, 48000), 2)

    except:
        print('error')
        inputnote = 0
   # print(inputnote)
   # if you want to use some other sount uncomment  line 187
  #  inputnote-=24000
    if inputnote > frequencies[len(tunerNotes)-1]:
        continue

    # not interested in notes below the notes list
    if inputnote < frequencies[0]:
        continue
    if signal_level > soundgate:
        continue

    targetnote = closest_value_index(frequencies, round(inputnote, 2))
    print(tunerNotes[frequencies[targetnote]])
