
# coding: utf-8

# In[19]:

import NeuralStyleTransfer
global i=0


# In[ ]:

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np 
from sys import argv
import torchvision.transforms as transforms
import copy
import librosa
import IPython

class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.cnn1 = nn.Conv1d(in_channels=1025, out_channels=4096, kernel_size=3, stride=1, padding=1)
        
        def forward(self, x):
            out = self.cnn1(x)
            out = out.view(out.size(0),-1)
            return out


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c = input.size() 
        features = input.view(a * b, c)
        G = torch.mm(features, features.t())  
        return G.div(a * b * c)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

if __name__ == '__main__':
    script, content_audio_name , style_audio_name = argv

    N_FFT=2048
    def read_audio_spectum(filename):
        x, fs = librosa.load(filename, duration=4) 
        librosa.output.write_wav('0'+filename, x, fs)
        S = librosa.stft(x, N_FFT)
        p = np.angle(S)
        S = np.log1p(np.abs(S))  
        return S, fs

    style_audio, style_sr = read_audio_spectum('melody4.mp3')
    content_audio, content_sr = read_audio_spectum('Content.wav')

    if(content_sr == style_sr):
        print('Sampling Rates are same')
    else:
        print('Sampling rates are not same')
        exit()

    num_samples=style_audio.shape[1]
    
    style_audio = style_audio.reshape([1,1025,num_samples])
    content_audio = content_audio.reshape([1,1025,num_samples])


    if torch.cuda.is_available():
        style_float = Variable((torch.from_numpy(style_audio)).cuda())
        content_float = Variable((torch.from_numpy(content_audio)).cuda())
        print('cuda')
    else:
        style_float = Variable(torch.from_numpy(style_audio))
        content_float = Variable(torch.from_numpy(content_audio))
        print('cpu')


    cnn = CNNModel()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    style_layers_default = ['conv_1']

    style_weight=2500

    def get_style_model_and_losses(cnn,
                                   style_float,style_weight=style_weight, 
                                   style_layers=style_layers_default): 
        
        cnn = copy.deepcopy(cnn)
        style_losses = []
        model = nn.Sequential() 
        gram = GramMatrix()  
        if torch.cuda.is_available():
            model = model.cuda()
            gram = gram.cuda()

        name = 'conv_1'
        model.add_module(name, cnn.cnn1)
        if name in style_layers:
            target_feature = model(style_float).clone()
            target_feature_gram = gram(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            model.add_module("style_loss_1", style_loss)
            style_losses.append(style_loss)



        return model, style_losses


    in_fl = content_float.clone()
    #in_fl = Variable(torch.randn(content_float.size())).type(torch.FloatTensor)

    learning_rate_initial = 0.01

    def get_input_param_optimizer(in_fl):
        input_param = nn.Parameter(in_fl.data)
        #optimizer = optim.Adagrad([input_param], lr=learning_rate_initial, lr_decay=0.0001,weight_decay=0)
        optimizer = optim.Adam([input_param], lr=learning_rate_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return input_param, optimizer

    steps_n= 2500

    def run_style_transfer(cnn, style_float, in_fl, steps_n=steps_n, style_weight=style_weight): #STYLE WEIGHT, steps_n
        print('Building the style transfer model..')
        model, style_losses= get_style_model_and_losses(cnn, style_float, style_weight)
        input_param, optimizer = get_input_param_optimizer(in_fl)
        print('Optimizing..')
        run = [0]
        i = 0
        while run[0] <= steps_n:
            if steps_n//100 == 0:
                print(i)
            i += 1
            def closure():
                # correct the values of updated input image
                input_param.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_param)
                style_score = 0

                for sl in style_losses:
                    #print('sl is ',sl,' style loss is ',style_score)
                    style_score += sl.backward()

                run[0] += 1
                if run[0] % 100 == 0:
                    print('Style Loss : {:8f}'.format(style_score.data[0])) #CHANGE 4->8 
                    print()

                return style_score


            optimizer.step(closure)
        input_param.data.clamp_(0, 1)
        return input_param.data
    
    output = run_style_transfer(cnn, style_float, in_fl)
    if torch.cuda.is_available():
        output = output.cpu()

    output = output.squeeze(0)
    output = output.numpy()
    
    N_FFT=2048
    a = np.zeros_like(output)
    a = np.exp(output) - 1

    # This code is supposed to do phase reconstruction
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        S = a * np.exp(1j*p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    #OUTPUT_FILENAME = 'output1D_4096_iter'+str(steps_n)+'_c'+content_audio_name+'_s'+style_audio_name+'_sw'+str(style_weight)+'_k3s1p1.wav'
    librosa.output.write_wav('out{}.wav'.format(i), x, style_sr)
    i +=1
    print('DONE...')


# In[41]:

IPython.display.Audio("out499.wav")


# In[ ]:




# In[ ]:



