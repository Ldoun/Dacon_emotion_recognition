import torch.nn as nn
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

class HuggingFace(nn.Module):
    def __init__(self, args, input_size, output_size) -> None:
        super().__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(args.pretrained_model, num_labels=output_size)

    def forward(self, x):
        return self.model(x).logits
    