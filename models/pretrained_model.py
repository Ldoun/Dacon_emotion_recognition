import torch.nn as nn
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

class HuggingFace(nn.Module):
    def __init__(self, args, input_size, output_size):
        super().__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(args.pretrained_model, num_labels=output_size, ignore_mismatched_sizes=True)
        #model In use: wav2vec2, ast, conformer, hubert, unispeech, wavlm, data2vec, sew

    def forward(self, x):
        return self.model(x).logits #call .logtis for Compatibility