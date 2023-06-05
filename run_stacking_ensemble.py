import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

from supervised.automl import AutoML


parser = argparse.ArgumentParser()
parser.add_argument("--name", default= None)

parser.add_argument("--train_csv", required=True)
parser.add_argument("--stacking_input", required=True)
parser.add_argument("--test_stacking_input", required=True)
parser.add_argument("--submission", required=True)

parser.add_argument("--target_col", required=True)
parser.add_argument("--drop", nargs='*')

parser.add_argument("--seed", type=int, default=72)
parser.add_argument("--eval_metric", default='accuracy')

args = parser.parse_args()

train_data = pd.read_csv(args.train_csv) #OOF Predictions of models
stacking_input = pd.read_csv(args.stacking_input)
test_staking_input = pd.read_csv(args.test_stacking_input) #Test Set Predictions of models

if args.drop is not None:
    train_x = stacking_input.drop(columns=args.drop + [args.target_col], axis = 1)
    test_x = test_staking_input.drop(columns=args.drop, axis = 1)
else:
    train_x = stacking_input
    test_x = test_staking_input
    
train_y = train_data[args.target_col]

automl = AutoML(mode="Compete", eval_metric=args.eval_metric, total_time_limit = 60 * 60 * 5, random_state=args.seed, results_path=os.path.join('result', args.name))
automl.fit(train_x, train_y)
#Use the AutoML to get maximum performance given OOF predictions of Models

pred = automl.predict(test_x) #predict using Test data
submission = pd.read_csv(args.submission)
submission[args.target_col] = pred
submission.to_csv(f"{automl.results_path}_submission.csv", index=False)