Dacon Voice Emotion Recognition AI Competition

[Korean](README.md)

The model and args that I used

For script wit out batch size argument, the batch size will be fixed by auto_batch_size.py befor the training session starts
```
python main.py --model RNN --batch_size 1024
python main.py --model LSTM --batch_size 1024
python main.py --model Transformer --batch_size 512 --lr 1e-4
python main.py --model HuggingFace --pretrained_model "MIT/ast-finetuned-audioset-10-10-0.4593" --batch_size 16 --lr 1e-5 --patience 10
python main.py --model HuggingFace --pretrained_model "facebook/wav2vec2-base" --batch_size 16 --lr 1e-5 --patience 10
python main.py --model HuggingFace --pretrained_model "facebook/wav2vec2-conformer-rope-large-960h-ft" --lr 1e-5 --patience 10
python main.py --model HuggingFace --pretrained_model "microsoft/wavlm-large" --lr 1e-5 --patience 10 --num_workers 4 --batch_size 6 
python main.py --model HuggingFace --pretrained_model "facebook/wav2vec2-conformer-rel-pos-large" --lr 1e-5 --patience 10 --num_workers 4
python main.py --model HuggingFace --pretrained_model "facebook/hubert-xlarge-ll60k" --lr 1e-5 --patience 10
python main.py --model HuggingFace --pretrained_model "microsoft/unispeech-sat-large" --lr 1e-5 --patience 10
python main.py --model HuggingFace --pretrained_model "microsoft/unispeech-large-1500h-cv" --lr 1e-5 --patience 10 --num_workers 4
python main.py --model HuggingFace --pretrained_model "facebook/data2vec-audio-large" --lr 1e-5 --patience 10 --num_workers 4
python main.py --model HuggingFace --pretrained_model "asapp/sew-mid-100k" --lr 1e-5 --patience 10 --num_workers 4 --batch_size 16
```

scipt when the model stopped unexpectedly
```
python main.py --model HuggingFace --pretrained_model "microsoft/wavlm-large" --lr 1e-5 --patience 10 --num_workers 4 --batch_size 6 --continue_train 5 --continue_from_folder result/HuggingFace_2_needed
```

Staking Ensemble using AutoML(Applied Stacking Ensemble to HuggingFace model only)
```
python make_input_for_stacking.py --n_mfcc 32
python run_stacking_ensemble.py --name stacking_input_with_mfcc --train_csv ../data/train.csv --stacking_input ./result/train_stacking_with_mfcc.csv --test_stacking_input ./result/test_stacking_with_mfcc.csv --submission ../data/sample_submission.csv --target_col label
```