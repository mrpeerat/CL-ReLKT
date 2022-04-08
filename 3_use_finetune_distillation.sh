
#!/bin/bash 
corpus='XORQA'
corpus_mode='train'
top_start=0
top_end=0
batch_size=16
num_epoch=30
gpu_device=5
use_mode='finetuned_USE_XORQA_train_en-ru_ko_ja_fi_top0-0_para_finetuned_USE_XORQA_train_en-ru_ko_ja_fi_top1-3_para_0.27margin_2WarmStep_cosLoss_RegFalse_FCFalse_small_0.27_1BatchUpdate_2hard_3shard_cos_HardUpdateFalse_SemiHardUpdateFalse_RegFalse'

languages='ru_ko_ja'
# languages='fi'

learning_rate="1e-3"
# mse_factor_q_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4") # finetuning range
# mse_factor_d_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_qd_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")

# sup setting
mse_factor_q_all=("1e-1")
mse_factor_d_all=("1e0")
mse_factor_qd_all=("1e-1")

# # unsup setting
# mse_factor_q_all=("1e0")
# mse_factor_d_all=("1e0")
# mse_factor_qd_all=("1e-1")

mse_factor=1000
shuffle="True"
replace="True"


for mse_factor_q in "${mse_factor_q_all[@]}"
do
    for mse_factor_d in "${mse_factor_d_all[@]}"
    do
        for mse_factor_qd in "${mse_factor_qd_all[@]}"
        do
            python use_finetune_distillation.py -mse_factor $mse_factor -replace $replace -mse_factor_q $mse_factor_q -mse_factor_d $mse_factor_d -mse_factor_qd $mse_factor_qd -shuffle $shuffle -learning_rate $learning_rate -languages $languages -num_epoch $num_epoch -batch_size $batch_size -corpus $corpus -corpus_mode $corpus_mode -top_start $top_start -top_end $top_end -use_mode $use_mode -gpu_device $gpu_device
        done
    done
done

##############################################################################################

#!/bin/bash 
corpus='XQUAD'
corpus_mode='train'
top_start=0
top_end=0
batch_size=16
num_epoch=30
gpu_device=7
use_mode='finetuned_USE_XQUAD_train_en-ar_de_es_ro_ru_th_zh_tr_top0-0_para_finetuned_USE_XQUAD_train_en-ar_de_es_ro_ru_th_zh_tr_top1-3_para_0.24margin_2WarmStep_cosLoss_RegFalse_FCFalse_0.24_1BatchUpdate_1hard_1shard_cos_HardUpdateFalse_SemiHardUpdateFalse_RegFalse'


languages='ar_de_es_ru_th_zh_tr'
# languages='ro_el_hi_vi'

learning_rate="1e-3"
# mse_factor_q_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_d_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_qd_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")


# sup setting
mse_factor_q_all=("1e-3")
mse_factor_d_all=("1e-1")
mse_factor_qd_all=("1e-2")

# unsup setting
# mse_factor_q_all=("1e-4")
# mse_factor_d_all=("1e-1")
# mse_factor_qd_all=("1e-2")

mse_factor=1000
shuffle="True"
replace="True"


for mse_factor_q in "${mse_factor_q_all[@]}"
do
    for mse_factor_d in "${mse_factor_d_all[@]}"
    do
        for mse_factor_qd in "${mse_factor_qd_all[@]}"
        do
            python use_finetune_distillation.py -mse_factor $mse_factor -replace $replace -mse_factor_q $mse_factor_q -mse_factor_d $mse_factor_d -mse_factor_qd $mse_factor_qd -shuffle $shuffle -learning_rate $learning_rate -languages $languages -num_epoch $num_epoch -batch_size $batch_size -corpus $corpus -corpus_mode $corpus_mode -top_start $top_start -top_end $top_end -use_mode $use_mode -gpu_device $gpu_device
        done
    done
done


#############################################################################################

#!/bin/bash 
corpus='MLQA'
corpus_mode='train'
top_start=0
top_end=0
batch_size=16
num_epoch=20
gpu_device=5
use_mode='finetuned_USE_MLQA_train_en-ar_de_es_zh_para_aligned_finetuned_USE_MLQA_train_en-ar_de_es_zh_top1-3_para_aligned_0.13margin_3WarmStep_cosLoss_1e-05LR_RegFalse_FCFalse_0.13_1BatchUpdate_1hard_1shard_cos_HardUpdateFalse_SemiHardUpdateFalse'


languages='ar_de_es_zh'
# languages='vi_hi'

learning_rate="5e-4"
# mse_factor_q_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_d_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_qd_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")

# sup setting
mse_factor_q_all=("1e-2")
mse_factor_d_all=("1e0")
mse_factor_qd_all=("1e-4")

# unsup setting
# mse_factor_q_all=("1e-2")
# mse_factor_d_all=("1e-2")
# mse_factor_qd_all=("1e-4")

mse_factor=1000
shuffle="True"
replace="True"


for mse_factor_q in "${mse_factor_q_all[@]}"
do
    for mse_factor_d in "${mse_factor_d_all[@]}"
    do
        for mse_factor_qd in "${mse_factor_qd_all[@]}"
        do
            python use_finetune_distillation_mlqa.py -mse_factor $mse_factor -replace $replace -mse_factor_q $mse_factor_q -mse_factor_d $mse_factor_d -mse_factor_qd $mse_factor_qd -shuffle $shuffle -learning_rate $learning_rate -languages $languages -num_epoch $num_epoch -batch_size $batch_size -corpus $corpus -corpus_mode $corpus_mode -top_start $top_start -top_end $top_end -use_mode $use_mode -gpu_device $gpu_device
        done
    done
done
