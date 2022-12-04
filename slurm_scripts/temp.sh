#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 10
#SBATCH -G 4
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -o abs_histbert_log.out

# retrain from historical data (histbert)
# bert_data_path='../data/historical_samples/bert_data/bert.pt_data/'
# model_data_path='../data/historical_samples/abstractive/basic_model_path/'
# log_file_path='../logs/bertabs_train_historical_data_from_pretrained.log'
# train_from_path='../data/baseline_model/bertextabs_cnndm_final_148000.pt'
# train_steps=200000

# 2: retrain from cnn data (histbert)
# bert_data_path='../data/cnn/bert_data/cnndm/cnndm.'
# model_data_path='../data/cnn/model_path/abstractive/'
# log_file_path='../logs/bertabs_train_cnn_data_from_pretrained.log'
# train_from_path='../data/baseline_model/bertextabs_cnndm_final_148000.pt'
# train_steps=200000

# 3: retrain from historical data - initialize with 2
# bert_data_path='../data/historical_samples/bert_data/bert.pt_data/'
# model_data_path='../data/historical_samples/abstractive/pre_model_path/'
# log_file_path='../logs/bertabs_train_historical_data_from_pretrained_hist.log'
# train_from_path='../data/cnn/model_path/abstractive/model_step_200000.pt'
# train_steps=250000

# 4: baseline retrain - retrain from historical data - histbert
bert_data_path='../data/historical_samples/bert_data/bert.pt_data/'
model_data_path='../data/historical_samples/model_path/abstractive/retrain_model_path/'
log_file_path='../logs/abstractive_retrain_model_path.log'
train_from_path='../data/cnn/model_path/abstractive_baseline/model_step_200000.pt'
train_steps=250000

cd /home/users/kmurugaraj/masterthesis/PreSumm/src
conda activate histsumm_gpu
python train.py -task abs -mode train -bert_data_path $bert_data_path -dec_dropout 0.2 -model_path $model_data_path -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps $train_steps -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file $log_file_path -model histbert -train_from $train_from_path
