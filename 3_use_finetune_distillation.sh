
#!/bin/bash 
corpus='XORQA'
corpus_mode='train'
top_start=0
top_end=0
batch_size=16
num_epoch=30
gpu_device=5
use_mode='teacher_model'

languages='ru_ko_ja'
# languages='fi'

learning_rate="1e-3"
# mse_factor_q_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4") # finetuning range
# mse_factor_d_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_qd_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")


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
use_mode='teacher_model'

languages='ar_de_es_ru_th_zh_tr'
# languages='ro_el_hi_vi'

learning_rate="1e-3"
# mse_factor_q_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_d_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_qd_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")

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
use_mode='teacher_model'

languages='ar_de_es_zh'
# languages='vi_hi'

learning_rate="5e-4"
# mse_factor_q_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_d_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")
# mse_factor_qd_all=("1e0" "1e-1" "1e-2" "1e-3" "1e-4")

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
