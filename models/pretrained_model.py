import torch.nn as nn
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

class Wav2Vec2(nn.Module):
    def __init__(self, args, input_size, output_size) -> None:
        super().__init__()
        self.model_name = args.pretrained_model
        self.model = AutoModelForAudioClassification.from_pretrained(self.pretrained_model, num_labels=output_size)

    def forward(self, x):
        return self.model(x)
    