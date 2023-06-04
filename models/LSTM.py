import torch
import torch.nn as nn

class LSTM(nn.Module): 
    def __init__(self, args, input_size, output_size):
        super().__init__()
        
        self.hidden_size = args.hidden
        self.model = nn.LSTM(
            input_size=input_size, hidden_size=self.hidden_size, num_layers=args.n_layer, 
            batch_first=True, dropout=args.drop_p, bidirectional=True
        ) #bi-directional LSTM
        self.linear = nn.Linear(2*args.hidden, output_size)

    def forward(self, audio):
        output, _  = self.model(audio)#(bs, timestep, 2*hidden)
        forward_o = output[:, -1, :self.hidden_size] #hidden state of last output of forward step
        backward_o = output[:, 0, self.hidden_size:] #hidden state of last output of backward step
        
        out = torch.cat([forward_o, backward_o], dim=1) #concat outputs of forward step and backward step
        return self.linear(out)
        
class RNN(nn.Module):
    def __init__(self, args, input_size, output_size):
        super().__init__()
        
        self.hidden_size = args.hidden
        self.model = nn.RNN(input_size=input_size, hidden_size=args.hidden, num_layers=args.n_layer, batch_first=True, 
                        dropout=args.drop_p, bidirectional=True, nonlinearity='tanh'
        ) #bi-directional RNN
        self.linear = nn.Linear(2*args.hidden, output_size)

    def forward(self, audio):
        output, _  = self.model(audio)#(bs, timestep, 2*hidden)
        forward_o = output[:, -1, :self.hidden_size] #hidden state of last output of forward step
        backward_o = output[:, 0, self.hidden_size:] #hidden state of last output of backward step
        
        out = torch.cat([forward_o, backward_o], dim=1) #concat outputs of forward step and backward step
        return self.linear(out)