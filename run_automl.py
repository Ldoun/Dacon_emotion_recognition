import os
import random
import argparse
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
from supervised.automl import AutoML
from sklearn.preprocessing import MinMaxScaler


from data import load_audio_mfcc
from config import args_for_audio, args_for_data


parser = argparse.ArgumentParser()
parser.add_argument("--name", default= None)
parser.add_argument("--target_col", required=True)

parser.add_argument("--seed", type=int, default=72)
parser.add_argument("--eval_metric", default='accuracy')

args_for_audio(parser)
args_for_data(parser)

args = parser.parse_args()

process_func = partial(load_audio_mfcc, 
            sr=args.sr, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length, n_mels=args.n_mels, n_mfcc=args.n_mfcc)
process_func = lambda x: np.mean(x.T, axis=1)

train_data = pd.read_csv(args.train)
train_data['path'] = train_data['path'].apply(lambda x: os.path.join(args.path, x))
test_data = pd.read_csv(args.test)
test_data['path'] = test_data['path'].apply(lambda x: os.path.join(args.path, x))

train_features = [process_func(file) for file in train_data['path']]
test_features = [process_func(file) for file in test_data['path']]

min_max_scaler = MinMaxScaler().fit(train_features)
train_features = min_max_scaler.transform(train_features)
test_features = min_max_scaler.transform(test_features)

train_mfcc_df = pd.DataFrame(train_features, columns=['mfcc_'+str(x) for x in range(1,args.n_mfcc+1)])
test_mfcc_df = pd.DataFrame(test_features, columns=['mfcc_'+str(x) for x in range(1,args.n_mfcc+1)])
  
train_y = train_data[args.target_col]

result_path = os.path.join(args.result_path, args.name)
automl = AutoML(mode="Compete", eval_metric=args.eval_metric, total_time_limit = 60 * 60 * 5, random_state=args.seed, results_path=result_path)
automl.fit(train_mfcc_df, train_y)

pred = automl.predict(test_mfcc_df)
submission = pd.read_csv(args.submission)
submission[args.target_col] = pred
submission.to_csv(os.path.join(result_path, "automl_submission.csv"), index=False)