import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module): #Custom Batch-First Positional Encoding modified based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] #will be broadcasted to match the target tensor size
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, args, input_size, output_size):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_size, args.drop_p) # PostionalEmbedding was not used because some of test set data lengh is longer than train set
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=args.nhead, dim_feedforward=args.dim_ff, dropout=args.drop_p, batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.n_layers)
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, audio):
        x = self.pos_encoder(audio)
        x = self.model(x)
        out = self.output_layer(x[:, 0, :]) #Similar to Bert(CLS), 1th time step output will be used for classification
        return out