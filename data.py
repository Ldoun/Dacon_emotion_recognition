import librosa
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import  Dataset

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
    y = pad_sequence(y, batch_first=True) 
    return x, y

class AudioDataSet(Dataset):
    def __init__(self, process_func, file_list, y=None):
        self.features = torch.tensor([process_func(file) for file in file_list], dtype=torch.float)
        self.min = self.features.min()
        self.max = self.features.max()
        
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.zeros(self.features.shape[0])

        self.scaler = None

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.scaler(self.features[index]), self.y[index]