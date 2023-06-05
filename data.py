import librosa
import numpy as np

import torch
from torch.utils.data import  Dataset
from torch.nn.utils.rnn import pad_sequence

def load_audio_mel_spectrogram(file, sr, n_fft, win_length, hop_length, n_mels): #compute mel spectrogram from audio file
    data, _ = librosa.load(file, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)
    power_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref = np.max)
    return power_mel_spectrogram  

def load_audio_mfcc(file, sr, n_fft, win_length, hop_length, n_mels, n_mfcc): #compute mfcc feature from audio file
    data, _ = librosa.load(file, sr=sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
    return mfcc.T

def load_audio(file, sr): #simply load audio
    data, _ = librosa.load(file, sr=sr)
    return data

def collate_fn(batch): #collate_fn for making batch input
    x, y = zip(*batch)
    x = pad_sequence(x, batch_first=True)
    y = torch.stack(y)
    return x, y

class AudioDataSet(Dataset):
    def __init__(self, process_func, file_list, y=None):
        self.features = [process_func(file) for file in file_list]
        self.min = min([features.min() for features in self.features]) #for min-max scaler
        self.max = max([features.max() for features in self.features]) #for min-max scaler
        audio_length = [len(feature) for feature in self.features]
        max_index = audio_length.index(max(audio_length))
        self.max_length_file = file_list.iloc[max_index] #will be used to calculate the appropriate batch_size
        
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.long)
        else:
            self.y = torch.zeros(len(self.features)) #dummy y for Compatibility

        self.scaler = None

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return torch.tensor(self.scaler(self.features[index]), dtype=torch.float), self.y[index]