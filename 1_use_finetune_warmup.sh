#!/bin/bash 
# + {}
corpus='XORQA' # XORQA, XQUAD, MLQA
corpus_mode='train'
context_mode='para'
top_start=1
top_end=3
batch_size=16
num_epoch=10
gpu_device=5
loss_mode='cos'
replace='True'
use_mode='small'

warup_steps_all=(3)
margin_all=(0.27) # 0.13 = MLQA, 0.24 = XQUAD, 0.27 = XORQA
learning_rate="1e-5"

# For new dataset, please finetune the margin
# margin_all=(0.1 0.11 0.12 0.13)
# margin_all=(0.14 0.15 0.16 0.17) 
# margin_all=(0.18 0.19 0.2 0.21)
# margin_all=(0.22 0.23 0.24 0.25)
# margin_all=(0.26 0.27 0.28 0.29)
# margin_all=(0.30 0.31 0.32 0.33)
# margin_all=(0.34 0.35 0.36 0.37)
# margin_all=(0.38 0.39 0.40 0.41)


# languages='ar_de_es_ro_ru_th_zh_tr_el_hi_vi' # for XQUAD
# languages='ru_ko_ja_fi' # for XORQA
# languages='ar_de_es_zh_vi_hi' # for MLQA

for margin in "${margin_all[@]}"
do
    for warup_steps in "${warup_steps_all[@]}"
    do # 1_use_finetune_warmup_mlqa.py for train MLQA
        python 1_use_finetune_warmup.py -learning_rate $learning_rate -use_mode $use_mode -replace $replace -loss_mode $loss_mode -languages $languages -warup_steps $warup_steps -margin $margin -num_epoch $num_epoch -batch_size $batch_size -corpus $corpus -corpus_mode $corpus_mode -context_mode $context_mode -top_start $top_start -top_end $top_end -gpu_device $gpu_device
    done
done
