import argparse

def args_for_data(parser):
    parser.add_argument('--sr', type=int, default=16000, help='sampling rate')

    #mel-spectrogram
    parser.add_argument('--n_fft', type=int, default=2048, help='length of the windowed signal after padding with zeros')
    parser.add_argument('--win_length', type=int, default=200, help='each frame of audio is windowed by window of length')
    parser.add_argument('--hop_length', type=int, default=160, help='hop length')
    parser.add_argument('--n_mels', type=int, default=128, help='output shape of spectrogram will be (n_mels, ts)')

def args_for_model(parser, model):
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='', help='name of the model')

    args_for_data(parser)

    args_, _ = parser.parse_known_args()
    args_for_model(parser, args_.model)

    args = parser.parse_args()
    return args
