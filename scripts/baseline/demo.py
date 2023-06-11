import os
import time
import numpy as np
import datetime
import tqdm
from sklearn import metrics
import pickle
import json
import essentia


import streamlit as st

import torch
import torch.nn as nn

from model import CNN
from argparse import ArgumentParser
from pydub import AudioSegment
from essentia.standard import *
from decimal import Decimal


class Demo(object):
    def __init__(self, config):
        # Training settings
        self.n_epochs = 249
        self.lr = 1e-4
        self.log_step = 500
        self.is_cuda = torch.cuda.is_available()
        self.model_save_path = config.model_save_path
        self.batch_size = config.batch_size
        self.tag_list = self.get_tag_list(config)
        if config.subset == 'all':
            self.num_class = 183
        elif config.subset == 'genre':
            self.num_class = 87
            self.tag_list = self.tag_list[:87]
        elif config.subset == 'instrument':
            self.num_class = 40
            self.tag_list = self.tag_list[87:127]
        elif config.subset == 'moodtheme':
            self.num_class = 56
            self.tag_list = self.tag_list[127:]
        elif config.subset == 'top50tags':
            self.num_class = 50
        self.model_fn = os.path.join(self.model_save_path, 'best_model_3105_unknownacc.pth')
        self.roc_auc_fn = 'roc_auc_'+config.subset+'_'+str(config.split)+'.npy'
        self.pr_auc_fn = 'pr_auc_'+config.subset+'_'+str(config.split)+'.npy'
        self.f1_fn = 'f1_'+config.subset+'_'+str(config.split)+'.npy'
        self.audio_path = './scripts/baseline/demo_file/audio/audio_file.mp3'
        self.npy_output =  './scripts/baseline/demo_file/npy/npy_output.npy'

        # Build model
        self.build_model()

    def build_model(self):
        # model and optimizer
        model = CNN(num_class=self.num_class)

        if self.is_cuda:
            self.model = model
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def to_var(self, x):
        if self.is_cuda:
            x = torch.tensor(x).cuda()
        return x
    
    def get_tag_list(self, config):
        if config.subset == 'top50tags':
            path = './scripts/baseline/tag_list_50.npy'
        else:
            path = './scripts/baseline/tag_list.npy'
        tag_list = np.load(path)
        return tag_list
    
    def load_audio(self, filename, sampleRate=12000, segment_duration=None):
        audio = MonoLoader(filename=filename, sampleRate=sampleRate)()
    
        if segment_duration:
            segment_duration = round(segment_duration*sampleRate)
            segment_start = (len(audio) - segment_duration) // 2
            segment_end = segment_start + segment_duration
        else:
            segment_start = 0
            segment_end = len(audio)

        if segment_start < 0 or segment_end > len(audio):
            raise ValueError('Segment duration is larger than the input audio duration')

        return audio[segment_start:segment_end]


    def melspectrogram(self, audio, 
                    sampleRate=12000, frameSize=512, hopSize=256, 
                    window='hann', zeroPadding=0, center=True,
                    numberBands=96, lowFrequencyBound=0, highFrequencyBound=None,
                    weighting='linear', warpingFormula='slaneyMel', 
                    normalize='unit_tri'):

        if highFrequencyBound is None:
            highFrequencyBound = sampleRate/2
        
        windowing = Windowing(type=window, normalized=False, zeroPadding=zeroPadding)
        spectrum = Spectrum()
        melbands = MelBands(numberBands=numberBands,
                            sampleRate=sampleRate,
                            lowFrequencyBound=lowFrequencyBound, 
                            highFrequencyBound=highFrequencyBound,
                            inputSize=(frameSize+zeroPadding)//2+1,
                            weighting=weighting,
                            normalize=normalize,
                            warpingFormula=warpingFormula,
                            type='power')
        amp2db = UnaryOperator(type='lin2db', scale=2)

        pool = essentia.Pool()
        for frame in FrameGenerator(audio, 
                                    frameSize=frameSize, hopSize=hopSize,
                                    startFromZero=not center):
            pool.add('mel', amp2db(melbands(spectrum(windowing(frame)))))

        return pool['mel'].T


    def analyze(self, audio_file = '', npy_file = '', full_audio = True):
        if full_audio:
        # Analyze full audio duration.
            segment_duration=None
        else:
        # Duration for the Choi's VGG model.
            segment_duration=5

        audio = self.load_audio(audio_file, segment_duration=segment_duration)
        mel = self.melspectrogram(audio)
        np.save(npy_file, mel, allow_pickle=False)
        return

    
    def main(self):
        # Read the JSON file
        json_file_path = './scripts/baseline/tag_list_genre.json'
        with open(json_file_path) as f:
            genre_names = json.load(f)


        # Title
        st.title('Classifier for Music Genre')
        st.write('This is a classifier for music genre. It is based on the [mtg-jamendo-dataset]')

        # Upload file
        st.header('Upload your music file')
        uploaded_file = st.file_uploader("Choose a file", type=['wav', 'mp3', 'm4a', 'ogg'])


        if uploaded_file is not None:
            st.write('File successfully uploaded')
            st.audio(uploaded_file)

            # Preprocessing
            audio_path = './scripts/baseline/demo_file/audio/audio_file.mp3'
            npy_output =  './scripts/baseline/demo_file/npy/npy_output.npy'

            if uploaded_file is not None:
                with open(audio_path, "wb") as file:
                    file.write(uploaded_file.read())
                st.success("Audio file saved successfully.")

            # Load model
            self.load(self.model_fn)
            self.model.eval()

            # Threshhold
            threshhold = st.text_input('Thresh hold for genre', '0.3')
            threshhold = float(threshhold)

            self.analyze(audio_path, npy_output, True)

            original_array = np.load(npy_output)
            new_shape = (96, 234)

            result_dict = {}

            total_prd_array = [0] * 87

            number_of_segment = int(original_array.shape[1]/234)

            for i in range(number_of_segment):

                segment_start = i * 234
                segment_end = (i + 1) * 234
                resized_array = original_array[:, segment_start:segment_end]

                np.save(npy_output, resized_array)

                # Load npy
                prd_array = []  # prediction
                npy = np.load(npy_output)

                # Predict
                x = self.to_var(npy)
                out = self.model(x)

                # Get prediction
                out = out.cpu().detach().cpu()
                out = out.numpy()
                prd_array.append(out)

                original_prd_array = prd_array[0][0][:87]
                
                # print('original_prd_array: \n', original_prd_array)

                signmoid_prd_array = [0 if value < threshhold else 1 for value in original_prd_array]

                # print('signmoid_prd_array \n',signmoid_prd_array)

                total_prd_array += signmoid_prd_array

                total_prd_array = [x + y for x, y in zip(total_prd_array, signmoid_prd_array)]

                # print('total_prd_array \n', total_prd_array)

            weighed_prd_array = [float(x)/number_of_segment * 100 for x in total_prd_array]
            
            result_dict = {genre: value for genre, value in zip(genre_names, weighed_prd_array)}

            genre_list = []

            for key, value in result_dict.items():
                if value > 0.4:
                    genre_list.append(key)

            st.write('Prediction: ', genre_list)

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mode', type=str, default='TRAIN')
parser.add_argument('--model_save_path', type=str, default='./scripts/baseline/models')
parser.add_argument('--audio_path', type=str, default='/home')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--subset', type=str, default='all')

config = parser.parse_args() 

demo = Demo(config)
demo.main()


