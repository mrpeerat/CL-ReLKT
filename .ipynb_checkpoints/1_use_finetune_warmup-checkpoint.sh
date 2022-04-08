#!/bin/bash 
# + {}
corpus='MLQA'
corpus_mode='train'
context_mode='para'
top_start=1
top_end=2
batch_size=16
num_epoch=100
gpu_device=5
loss_mode='cos'
replace='True'
use_mode='small'

warup_steps_all=(3)
dropout_rates=(0.2)
margin_all=(0.27)
fc_dimensions=(512)
learning_rate="1e-5"

# fc_dimensions=(512)
# dropout_rates=(0.1 0.15 0.2 0.25 0.3 0.35)
# warup_steps_all=(1 2 3)
# ./use_finetune_random_cldr.sh

# margin_all=(0.1 0.11)
# margin_all=(0.12 0.13)
# margin_all=(0.14 0.15)
# margin_all=(0.16 0.17)
# margin_all=(0.18 0.19)
# margin_all=(0.2 0.21)
# margin_all=(0.22 0.23)

# margin_all=(0.24 0.25)
# margin_all=(0.26 0.27)
# margin_all=(0.28 0.29)
# margin_all=(0.30 0.31)
# margin_all=(0.32 0.33)
# margin_all=(0.34 0.35)
# margin_all=(0.36 0.37)

# margin_all=(0.38 0.39)
# margin_all=(0.40 0.41)
# margin_all=(0.42 0.43)

# margin_all=(0.44 0.45)
# margin_all=(0.46 0.47)
# margin_all=(0.48 0.49)
# margin_all=(0.50 0.51)

# margin_all=(0.1 0.11 0.12 0.13)
# margin_all=(0.14 0.15 0.16 0.17) 

# margin_all=(0.18 0.19 0.2 0.21)
# margin_all=(0.22 0.23 0.24 0.25)

# margin_all=(0.26 0.27 0.28 0.29)
# margin_all=(0.30 0.31 0.32 0.33)

# margin_all=(0.34 0.35 0.36 0.37)
# margin_all=(0.38 0.39 0.40 0.41)

# languages='ru_ko_ja_fi'
# languages='ar_de_es_ro_ru_th_zh_tr_el_hi_vi'
# languages='ar_de_es_ro_ru_th_zh_tr'
languages='ar_de_es_zh'
# languages='en'

# python use_finetune_random_cldr_dropout_fc_mlqa.py

for margin in "${margin_all[@]}"
do
    for warup_steps in "${warup_steps_all[@]}"
    do
        for dropout_rate in "${dropout_rates[@]}"
        do
            for fc_dimension in "${fc_dimensions[@]}"
            do
                python use_finetune_random_cldr_mlqa.py -learning_rate $learning_rate -use_mode $use_mode -fc_dimension $fc_dimension -replace $replace -dropout_rate $dropout_rate -loss_mode $loss_mode -languages $languages -warup_steps $warup_steps -margin $margin -num_epoch $num_epoch -batch_size $batch_size -corpus $corpus -corpus_mode $corpus_mode -context_mode $context_mode -top_start $top_start -top_end $top_end -gpu_device $gpu_device
            done
        done
    done
done
