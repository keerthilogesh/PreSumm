#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 10
#SBATCH -G 4
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -o test_model.out

method=abs
bert_data_path='../data/cnn/bert_data/cnndm/cnndm.'
model_data_path='../data/cnn/model_path/abstractive_baseline/'
result_path='../logs/test_abs_baseline_20000.log'
model_path='../data/cnn/model_path/abstractive_baseline/model_step_200000.pt'

cd /home/users/kmurugaraj/masterthesis/PreSumm/src
conda activate histsumm_gpu
python train.py -task $method -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path $bert_data_path -model_path $model_data_path -sep_optim true -use_interval true -visible_gpus 0,1,2,3 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path $result_path -test_from $model_path