#!/bin/bash 
corpus='XORQA' # XORQA, XQUAD, MLQA
corpus_mode='train'
context_mode='doc'
top_start=0
top_end=0
batch_size=16
num_epoch=100
gpu_device=1
use_mode='small'
batch_update=1
loss_mode='cos' # cos, euc
hard_update='False' 
semi_hard_update='False'
replace='True'
dropout_rates=(0.31)
fc_dimensions=(512)

hard=2 # XORQA = 2, XQUAD = 1, MLQA = 1
semi_hard=3 # XORQA = 3, XQUAD = 1, MLQA = 1
margin_all=(0.27) # 0.13 = MLQA, 0.24 = XQUAD, 0.27 = XORQA

# languages='ar_de_es_ro_ru_th_zh_tr_el_hi_vi' # for XQUAD
# languages='ru_ko_ja_fi' # for XORQA
# languages='ar_de_es_zh_vi_hi' # for MLQA


for margin in "${margin_all[@]}"
do # use_finetune_teacher_mlqa.py for MLQA trianing
    python use_finetune_teacher.py -replace $replace -languages $languages -hard_update $hard_update -semi_hard_update $semi_hard_update -loss_mode $loss_mode -batch_update $batch_update -hard $hard -semi_hard $semi_hard -margin $margin -num_epoch $num_epoch -batch_size $batch_size -corpus $corpus -corpus_mode $corpus_mode -context_mode $context_mode -top_start $top_start -top_end $top_end -use_mode $use_mode -gpu_device $gpu_device
done



