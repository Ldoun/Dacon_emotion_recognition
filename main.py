import os
import logging
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import StratifiedKFold

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from config import get_args
from trainer import Trainer
import models as model_module
from utils import seed_everything
from data import load_audio_mfcc, AudioDataSet, collate_fn

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    device = torch.device('cuda:0')

    result_path = os.path.join(args.result_path, str(len(os.listdir(args.result_path))))
    os.makedirs(result_path)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    

    train_data = pd.read_csv(args.train)
    train_data['path'] = train_data['path'].apply(lambda x: os.path.join(args.path, x))
    test_data = pd.read_csv(args.test)
    test_data['path'] = test_data['path'].apply(lambda x: os.path.join(args.path, x))
    
    process_func = partial(load_audio_mfcc, 
            sr=args.sr, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length, n_mels=args.n_mels, n_mfcc=args.n_mfcc)
    
    input_size = args.n_mfcc
    output_size = 6

    test_result = []
    skf = StratifiedKFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True)
    for fold, (train_index, valid_index) in enumerate(skf.split(train_data['path'], train_data['label'])):
        fold_result_path = os.path.join(result_path, f'{fold+1}-fold')
        os.makedirs(fold_result_path)
        fold_logger = logger.getChild(f'{fold+1}-fold')
        fold_logger.handlers.clear()
        fold_logger.addHandler(logging.FileHandler(os.path.join(fold_result_path, 'log.log')))    
        fold_logger.info(f'start training of {fold+1}-fold')

        kfold_train_data = train_data.iloc[train_index]
        kfold_valid_data = train_data.iloc[valid_index]

        train_dataset = AudioDataSet(process_func=process_func, file_list=kfold_train_data['path'], y=kfold_train_data['label'])
        valid_dataset = AudioDataSet(process_func=process_func, file_list=kfold_valid_data['path'], y=kfold_valid_data['label'])
        scaler = lambda x:(x-train_dataset.min)/(train_dataset.max-train_dataset.min)
        train_dataset.scaler = scaler
        valid_dataset.scaler = scaler

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
        )

        model = getattr(model_module, args.model)(args, input_size, output_size).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        trainer = Trainer(train_loader, valid_loader, model, loss_fn, optimizer, device, args.patience, args.epochs, result_path, fold_logger, len(train_dataset), len(valid_dataset))
        trainer.train()

        test_dataset = AudioDataSet(process_func=process_func, file_list=test_data['path'], y=None)
        test_dataset.scaler = scaler
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
        )
        test_result.append(trainer.test(test_loader))

    test_result = np.sum(test_result, axis=0)
    prediction = pd.read_csv(args.submission)
    prediction['label'] = np.argmax(test_result, axis=-1)
    prediction.to_csv(os.path.join(result_path, 'prediction.csv'), index=False)
