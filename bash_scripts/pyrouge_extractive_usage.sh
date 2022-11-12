#!/usr/bin/env bash
# [-a (evaluate all systems)]
# [-c cf]
# [-d (print per evaluation scores)]
# [-e ROUGE_EVAL_HOME]
# [-h (usage)]
# [-H (detailed usage)]
# [-b n-bytes|-l n-words]
# [-m (use Porter stemmer)]
# [-n max-ngram]
# [-s (remove stopwords)]
# [-r number-of-samples (for resampling)]
# [-2 max-gap-length (if < 0 then no gap length limit)]
# [-3 <H|HM|HMR|HM1|HMR1|HMR2> (for scoring based on BE)]
# [-u (include unigram in skip-bigram) default no)]
# [-U (same as -u but also compute regular skip-bigram)]
# [-w weight (weighting factor for WLCS)]
# [-v (verbose)]
# [-x (do not calculate ROUGE-L)]
# [-f A|B (scoring formula)]
# [-p alpha (0 <= alpha <=1)]
# [-t 0|1|2 (count by token instead of sentence)]
# [-z <SEE|SPL|ISI|SIMPLE>]
# <ROUGE-eval-config-file> [<systemID>]
perl C:/Users/keert/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl -e C:/Users/keert/pyrouge/tools/ROUGE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m C:/Users/keert/AppData/Local/Temp/tmpvchkvico/rouge_conf.xml

-train_from
../data/baseline_model/bertextabs_cnndm_final_148000.pt

# Transformer abs
python train.py -mode train -accum_count 5 -batch_size 300 -bert_data_path ../data/cnn/bert_data/cnndm/cnndm. -dec_dropout 0.1 -log_file ../logs/abs_bert_cnndm_baseline -lr 0.05 -model_path ../data/cnn/abs_model_path_baseline/ -save_checkpoint_steps 2000 -seed 777 -sep_optim false -train_steps 200000 -use_bert_emb true -use_interval true -warmup_steps 8000  -visible_gpus 0,1,2,3 -max_pos 512 -report_every 50 -enc_hidden_size 512  -enc_layers 6 -enc_ff_size 2048 -enc_dropout 0.1 -dec_layers 6 -dec_hidden_size 512 -dec_ff_size 2048 -encoder baseline -task abs -model histbert
# bert abs
python train.py  -task abs -mode train -bert_data_path ../data/cnn/bert_data/cnndm/cnndm. -dec_dropout 0.2  -model_path ../data/cnn/abs_model_path_abs/ -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file ../logs/abs_bert_cnndm -model histbert
# bert ext abs
python train.py  -task abs -mode train -bert_data_path ../data/cnn/bert_data/cnndm/cnndm. -dec_dropout 0.2  -model_path ../data/cnn/abs_model_path_extabs/ -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file ../logs/ext_abs_bert_cnndm  -load_from_extractive EXT_CKPT -model histbert
