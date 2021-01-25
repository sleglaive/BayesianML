#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Simon Leglaive (simon.leglaive@inria.fr)
License agreement in LICENSE.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf 
import librosa
import os

wlen_sec=64e-3
hop_percent=0.25
fs=16000
  
wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
nfft = wlen
hop = np.int(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

files = librosa.util.find_files('.')

for wavfile in files:
    
    x, fs = sf.read(wavfile)    
    x = x[:,0]
    x = x/np.max(np.abs(x))
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, 
                                 win_length=wlen,
                                 window=win) # STFT
    
    time = np.arange(x.shape[0])/fs
    
    plt.figure(figsize=(10,4))
    plt.plot(time, x, 'k')
    plt.ylabel('amplitude')
    plt.xlabel('time (s)')
    plt.ylim([-1,1])
    plt.xlim([time[0],time[-1]])
    plt.yticks([-1,0,1])
    plt.tight_layout()
    
    plt.savefig(os.path.basename(wavfile)[:-4] + '_waveform.png')
    
    # plt.figure(figsize=(10,10))
    # librosa.display.specshow(librosa.power_to_db(np.abs(X)**2), sr=fs, hop_length=hop, x_axis='time', y_axis='hz')
    # plt.set_cmap('magma')
    # axes = plt.gca()
    # axes.set_ylim([0,4000])
    
    # plt.ylabel('frequency (Hz)')
    # plt.xlabel('time (s)')
    # plt.tight_layout()
    
    # plt.savefig(os.path.basename(wavfile)[:-4] + '_spectro.svg')
