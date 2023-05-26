from models.LSTM import LSTM, RNN
from models.transformer import Transformer

def args_for_rnn(parser):
    parser.add_argument('--hidden', type=int, default=256, help='hidden size of the rnn')
    parser.add_argument('--n_layer', type=int, default=2, help='number of the rnn layer')
    parser.add_argument('--drop_p', type=float, default=0.5, help='drop out')

def args_for_transformer(parser):
    parser.add_argument('--drop_p', type=float, default=0.5, help='drop out')
    parser.add_argument('--nhead', type=int, default=8, help='number of the head')
    parser.add_argument('--dim_ff', type=int, default=2048, help='feed forward network dimension')
    parser.add_argument('--n_layers', type=int, default=5, help='encoder layer for transformer')