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
from data import load_audio_mfcc, AudioDataSet, collate_fn, load_audio
from auto_batch_size import max_gpu_batch_size

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    if args.continue_train > 0:
        result_path = args.continue_from_folder
    else:
        result_path = os.path.join(args.result_path, args.model+'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    #logger to log result of every output

    train_data = pd.read_csv(args.train)
    train_data['path'] = train_data['path'].apply(lambda x: os.path.join(args.path, x))
    test_data = pd.read_csv(args.test)
    test_data['path'] = test_data['path'].apply(lambda x: os.path.join(args.path, x))
    #fix path based on the data dir

    if args.model == 'HuggingFace':
        process_func = partial(load_audio, sr=args.sr)
        extractor = model_module.AutoFeatureExtractor.from_pretrained(args.pretrained_model)
        #extract feature information using pretrained model's feature exractor
        extractor = partial(extractor, sampling_rate=args.sr, return_tensors='np') #for Compatibility: var name-> scaler, return_tensors -> np
        #using partial to fix argument of the function
        scaler = lambda x: extractor(x).input_values.squeeze(0)
        input_size = None
    else:
        process_func = partial(load_audio_mfcc, 
            sr=args.sr, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length, n_mels=args.n_mels, n_mfcc=args.n_mfcc)
        #using partial to fix argument of the function
        scaler = None
        input_size = args.n_mfcc
    
    output_size = 6

    test_result = np.zeros([len(test_data), output_size])
    skf = StratifiedKFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True) #Using StratifiedKFold for cross-validation
    prediction = pd.read_csv(args.submission)
    output_index = [f'{i}' for i in range(0, output_size)]
    stackking_input = pd.DataFrame(columns = output_index, index=range(len(train_data))) #dataframe for saving OOF predictions

    if args.continue_train > 0:
        prediction = pd.read_csv(os.path.join(result_path, 'sum.csv'))
        test_result = prediction[output_index].values
        stackking_input = pd.read_csv(os.path.join(result_path, f'for_stacking_input.csv'))
    
    for fold, (train_index, valid_index) in enumerate(skf.split(train_data['path'], train_data['label'])): #by skf every fold will have similar label distribution
        if args.continue_train > fold+1:
            logger.info(f'skipping {fold+1}-fold')
            continue
        fold_result_path = os.path.join(result_path, f'{fold+1}-fold')
        os.makedirs(fold_result_path)
        fold_logger = logger.getChild(f'{fold+1}-fold')
        fold_logger.handlers.clear()
        fold_logger.addHandler(logging.FileHandler(os.path.join(fold_result_path, 'log.log')))    
        fold_logger.info(f'start training of {fold+1}-fold')
        #logger to log current n-fold output

        kfold_train_data = train_data.iloc[train_index]
        kfold_valid_data = train_data.iloc[valid_index]

        train_dataset = AudioDataSet(process_func=process_func, file_list=kfold_train_data['path'], y=kfold_train_data['label'])
        valid_dataset = AudioDataSet(process_func=process_func, file_list=kfold_valid_data['path'], y=kfold_valid_data['label'])
        if scaler is None: #if model does not belong to HuggingFace -> use min-max scaler to scale data
            scaler = lambda x:(x-train_dataset.min)/(train_dataset.max-train_dataset.min)
        train_dataset.scaler = scaler
        valid_dataset.scaler = scaler

        model = getattr(model_module , args.model)(args, input_size, output_size).to(device) #make model based on the model name and args
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.batch_size == None: #if batch size is not defined -> calculate the appropriate batch size
            args.batch_size = max_gpu_batch_size(device, process_func, logger, model, loss_fn, train_dataset.max_length_file)
            model = getattr(model_module , args.model)(args, input_size, output_size).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
        )

        trainer = Trainer(train_loader, valid_loader, model, loss_fn, optimizer, device, args.patience, args.epochs, fold_result_path, fold_logger, len(train_dataset), len(valid_dataset))
        trainer.train() #start training

        test_dataset = AudioDataSet(process_func=process_func, file_list=test_data['path'], y=None)
        test_dataset.scaler = scaler
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
        ) #make test data loader
        test_result += trainer.test(test_loader) #softmax applied output; accumulate test prediction of current fold model
        prediction[output_index] = test_result
        prediction.to_csv(os.path.join(result_path, 'sum.csv'), index=False) 
        
        stackking_input.loc[valid_index, output_index] = trainer.test(valid_loader) #use the validation data(hold out dataset) to make input for Stacking Ensemble model(out of fold prediction)
        stackking_input.to_csv(os.path.join(result_path, f'for_stacking_input.csv'), index=False)

prediction['label'] = np.argmax(test_result, axis=-1) #use the most likely results as my final prediction
prediction.drop(columns=output_index).to_csv(os.path.join(result_path, 'prediction.csv'), index=False)