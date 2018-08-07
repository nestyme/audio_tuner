
# coding: utf-8

# In[33]:

get_ipython().system(' pip install python_speech_features')


# In[432]:

import librosa
import Constants
import statistics as stats
import numpy as np
from copy import copy, deepcopy
import python_speech_features
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from pydub import AudioSegment
import librosa.display
from PIL import Image
import glob
import IPython.display
import pandas as pd
import soundfile as sf


prepro = True  # if True, run `python prepro.py` first before running `python train.py`.

vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz''' # ␀: Padding ␃: End of Sentence

    # data
data = "/data/private/voice/LJSpeech-1.0"
test_data = 'test_sents.txt'
ref_audio = 'ref1/*.wav'
Tx = 188 # Fixed length of text length.

    # signal processing
sr = 22050 # Sample rate.
n_fft = 2048 # fft points (samples)
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 50 # Number of inversion iterations
preemphasis = .97 # or None
max_db = 100
ref_db = 20

# model
embed_size = 256 # alias = E
encoder_num_banks = 16
decoder_num_banks = 8
num_highwaynet_blocks = 4
r = 5 # Reduction factor.
dropout_rate = .5

# training scheme
lr = 0.001 # Initial learning rate.
logdir = "logdir"
sampledir = 'samples'
batch_size = 32
num_iterations = 1000000
# In[248]:

input_length = 16000*5
batch_size = 32
n_mels = 320
sr=22050


# In[515]:

def Read_Audio(filename,duration=None):
    if not(duration == None):
        audio,sr=librosa.load(filename, duration=duration)
    else:
        audio,sr=librosa.load(filename)
    r=librosa.stft(audio)
    plt.figure(figsize=(15,3))
    plt.plot(audio)
    plt.title('Input audio')
    plt.grid()
    print('========================')
    plt.figure(figsize=(15,3))
    plt.plot(r)
    plt.title('Input audio`s spectrum')
    plt.grid()
    return audio,sr,r


def Write_Audio(audio, filename, sr):
    librosa.output.write_wav(filename,audio, sr=sr)


# In[529]:

def griffin_lim(spectrogram, n_iter=n_iter):
    x_best = deepcopy(spectrogram)
    for i in range(n_iter):
        x_t = librosa.istft(x_best,
                            hop_length,
                            win_length=win_length,
                            window="hann")
        est = librosa.stft(x_t,
                           n_fft,
                           hop_length,
                           win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        x_best = spectrogram * phase
    x_t = librosa.istft(x_best,
                        hop_length,
                        win_length=win_length,
                        window="hann")
    y = np.real(x_t)
    return y
def decompose(filename, draw_plots=False):
    audio,sr,r=Read_Audio(filename)
    S_full, phase = librosa.magphase(librosa.stft(audio))
    idx = slice(*librosa.time_to_frames([0, 5], sr=sr))
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(0.25, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
    if draw_plots == True:
        plt.title('Full spectrum')
        plt.colorbar()

        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                                 y_axis='log', sr=sr)
        plt.title('Back')
        plt.colorbar()
        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                                 y_axis='log', x_axis='time', sr=sr)
        plt.title('Front')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
   
    length = len(audio)
    back_voice = librosa.effects.time_stretch(griffin_lim(S_background), 0.5)
    front_voice = librosa.effects.time_stretch(griffin_lim(S_foreground), 0.5)
    back_voice = back_voice[:length]
    front_voice = front_voice[:length]
    #front_voice = librosa.effects.pitch_shift(front_voice, sr, n_steps=1,bins_per_octave=10)
    librosa.output.write_wav('back.wav',back_voice,sr=sr,norm=True)
    librosa.output.write_wav('front.wav',front_voice,sr=sr)
    return front_voice, back_voice, sr, audio


# In[530]:

#front, back, sr, audio = decompose('Audio/speech_0014_slice_1.wav')


# In[531]:

def eliminate_escapes(audio):
    tmp=copy(audio)
    print(max)
    median=abs(stats.median(audio))
    print('1 step')
    i=0
    while i < len(audio)-3:
        try:
            cnt = 0
            j = i-2
            while abs(audio[j-1])<median*0.02:
                j += 1
                if j>90000:
                    r=i+j
                    audio = np.concatenate((audio[:i],audio[r:]), axis=0)
                    i += j
                    continue
        except IndexError:
            print('oops_2')
            break
        i += 1
        #pbar.update(1)
    #pbar.close()
    return audio
            
def combine(texture, style):
    #tuned_audio = librosa.effects.pitch_shift(front_new, sr, n_steps=0,bins_per_octave=12)
    texture = AudioSegment.from_file("back.wav")
    style = AudioSegment.from_file("front.wav")
    tmp='output_audio_''{}.wav'.format('5')
    combined = sound.overlay(texture-10,gain_during_overlay=True)
    combined=combined[:len(style)]
    combined.export("Result.ogg")
    return librosa.load('Result.ogg')

# In[532]:

#new_audio=eliminate_escapes(audio)
#librosa.output.write_wav('new.wav',new_audio,sr=sr,norm=True)


# In[533]:

#new_front = eliminate_escapes(front_voice)


# In[534]:

#IPython.display.Audio('new.wav', rate=sr)


# In[535]:

#front_new, back_new, sr_new, audio_new = decompose('new.wav')


# In[536]:

#IPython.display.Audio(front_new, rate=sr)


# In[537]:

#librosa.util.normalize(front_voice, norm=1, fill=True)
#librosa.output.write_wav('back.wav',back_voice,sr=sr,norm=True)
#librosa.output.write_wav('front.wav',front_voice,sr=sr,norm=True)


# In[538]:

#IPython.display.Audio('front.wav', rate=sr)


# In[ ]:

#IPython.display.Audio(front_voice, rate=sr)
#tuned_audio = librosa.effects.pitch_shift(front_voice, sr, n_steps=4,bins_per_octave=12)
#texture = texture = AudioSegment.from_file("back.wav")
#tmp='output_audio_''{}.wav'.format('5')
#sf.write(tmp, tuned_audio, sr,format='wav')
#sound = AudioSegment.from_file(tmp)+7
#os.remove(tmp)
#combined = sound.overlay(texture,gain_during_overlay=True)
#combined=combined[:len(sound)]
#combined.export("Result_""{}.ogg".format('5'), format='ogg')


# In[ ]:

#combined


# In[100]:

#IPython.display.Audio(back_voice, rate=sr)


# In[74]:

#back_voice = griffin_lim(back)


# In[ ]:




# In[ ]:



