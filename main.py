import os
import logging
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from config import get_args
from trainer import Trainer
from utils import seed_everything
from data import load_audio, AudioDataSet, collate_fn

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    result_path = os.path.join(args.result_path, len(os.listdir(args.result_path)))
    os.makedirs(result_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    

    train_data = pd.read_csv(args.train)
    train_data['path'].apply(lambda x: os.path.join(args.path, x))
    test_data = pd.read_csv(args.test)
    test_data['path'].apply(lambda x: os.path.join(args.path, x))
    
    process_func = partial(load_audio, 
            sr=args.sr, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length, n_mels=args.n_mels)
    

    test_result = []
    skf = StratifiedKFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True)
    for fold, (train_index, valid_index) in enumerate(skf.split(train_data['path'], train_data['y'])):
        fold_result_path = os.path.join(result_path, f'{fold+1}-fold')
        os.makedirs(fold_result_path)
        fold_logger = logger.getChild()
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

        model = None
        loss_fn = None
        optimizer = None

        trainer = Trainer(train_loader, valid_loader, model, loss_fn, optimizer, fold_result_path, fold_logger)
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
