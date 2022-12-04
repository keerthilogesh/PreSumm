#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -G 2
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH -o ext_histbert_new.out

# retrain from historical data (histbert)
bert_data_path='../data/historical_samples_new/bert_data/bert.pt_data/'
model_data_path='../data/historical_samples_new/extractive/basic_model_path/'
log_file_path='../logs/train_historical_data_from_pretrained.log'
train_from_path='../data/baseline_model/bertext_cnndm_transformer.pt'
train_steps=50000

# 2: retrain from cnn data (histbert)
# bert_data_path='../data/cnn/bert_data/cnndm/cnndm.'
# model_data_path='../data/cnn/model_path/extractive/'
# log_file_path='../logs/bertext_train_cnn_data_from_pretrained.log'
# train_from_path='../data/baseline_model/bertext_cnndm_transformer.pt'
# train_steps=50000

# 3: retrain from historical data - initialize with 2
#bert_data_path='../data/historical_samples_new/bert_data/bert.pt_data/'
#model_data_path='../data/historical_samples_new/extractive/pre_model_path/'
#log_file_path='../logs/bertext_train_historical_data_from_pretrained_hist.log'
#train_from_path='../data/cnn/model_path/extractive/model_step_50000.pt'
#train_steps=70000

# 4: baseline retrain - retrain from historical data - histber
# bert_data_path='../data/historical_samples_new/bert_data/bert.pt_data/'
# model_data_path='../data/historical_samples_new/extractive/retrain_model_path/'
# log_file_path='../logs/bertext_train_cnn_data_from_pretrained.log'
# train_from_path='../data/cnn/model_path/extractive_baseline/model_step_50000.pt'
# train_steps=70000

cd /home/users/kmurugaraj/masterthesis/PreSumm/src
conda activate histsumm_gpu
python train.py -task ext -mode train -bert_data_path $bert_data_path -ext_dropout 0.1 -model_path $model_data_path -lr 2e-3 -visible_gpus 0,1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 300 -train_steps $train_steps -accum_count 2 -log_file $log_file_path -use_interval true -warmup_steps 10000 -max_pos 512 -model histbert -train_from $train_from_path
