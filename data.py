import librosa
import numpy as np

import torch
from torch.utils.data import  Dataset
from torch.nn.utils.rnn import pad_sequence

def load_audio_mel_spectrogram(file, sr, n_fft, win_length, hop_length, n_mels):
    data, _ = librosa.load(file, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)
    power_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref = np.max)
    return power_mel_spectrogram  

def load_audio_mfcc(file, sr, n_fft, win_length, hop_length, n_mels, n_mfcc):
    data, _ = librosa.load(file, sr=sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
    return mfcc

def collate_fn(batch):
    x, y = zip(*batch)
    x = pad_sequence(x, batch_first=True)
    y = torch.cat(y)
    return x, y

class AudioDataSet(Dataset):
    def __init__(self, process_func, file_list, y=None):
        self.features = [process_func(file) for file in file_list]
        self.min = min([features.min() for features in self.features])
        self.max = max([features.max() for features in self.features])
        
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.long)
        else:
            self.y = torch.zeros(len(self.features))

        self.scaler = None

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return torch.tensor(self.scaler(self.features[index]), dtype=torch.float), self.y[index]