import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio

class audio_Dataset(Dataset):
    def __init__(self, dataset, train_file, classes_names, transforms=None):
        super().__init__()
        self.transforms = transforms
        
        self.root_dir = os.getcwd()
        self.dataset = dataset.reset_index(drop=True)
        self.train_file = train_file
        self.classes_names = classes_names
        self.label_dict = self.create_label_dict()
        self.input_length = 500
        
        self.wav_labels = self.one_hot_encoding()
        
        self.to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                            sample_rate=16000, n_fft=500, n_mels=64,
                            hop_length=160, f_min=0, f_max=8000)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        fname = self.dataset["fname"][idx]
        wav_path = os.path.join(self.root_dir, self.train_file, fname)
        data, sr = torchaudio.load(wav_path)
        
        if self.transforms is not None:
            data = self.transforms(samples=data.numpy(), sample_rate=sr)
            data = torch.tensor(data)
            
        data = torchaudio.transforms.Resample(sr, 16000)(data)
        
        mel_spec = self.to_mel_spectrogram(data)            
        mel_spec = torchaudio.transforms.TimeMasking(time_mask_param=80)(mel_spec)
        #mel_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)(mel_spec)
        
        log_mel_spec = (mel_spec + torch.finfo(torch.float).eps).log()
        mel_data = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + np.finfo(np.float64).eps)
        mel_data = self.data_padding(mel_data, self.input_length)
        mel_data = torch.Tensor(mel_data)
        wav_label = self.wav_labels[idx]
        
        return mel_data, wav_label
    
    def create_label_dict(self):
        label_dict = {}
        for i, label_name in enumerate(self.classes_names):
            label_dict[label_name] = i
        
        return label_dict        
    
    def one_hot_encoding(self):
        label_encoder = LabelEncoder()
        data_encoder = self.dataset.copy()
        label_encoder.fit(self.classes_names)
        data_encoder['label'] = label_encoder.transform(self.dataset['label'])

        classes_number = pd.DataFrame([i for i in range(len(self.classes_names))])
        data_encoder = pd.DataFrame(data_encoder['label'])
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoder.fit(classes_number)
        wav_labels = one_hot_encoder.transform(data_encoder).toarray()

        return wav_labels
        
    def data_padding(self, data, input_length):
        if data.shape[-1] > input_length:
            max_offset = data.shape[-1] - input_length
            offset = np.random.randint(max_offset)
            data = data[:, :, offset:(input_length+offset)]

        else:
            max_offset = input_length - data.shape[-1]
            offset = max_offset//2
            data = np.pad(data, ((0, 0), (0, 0), (offset, max_offset - offset)), "constant")
            
        return data
