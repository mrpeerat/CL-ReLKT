#!/usr/bin/env python
# coding: utf-8

import argparse_config
arg_config = argparse_config.arg_convert()
import os
os.environ['TFHUB_CACHE_DIR']='/workspace/sentence-embedding/use-model'
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = arg_config.gpu_device
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config) 

import tensorflow_hub as hub
import tensorflow_text
from tensorflow.keras.layers import Layer,Input,Dense,Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import evaluate as eval_fnc

tf.config.experimental_run_functions_eagerly(True)


def USE_distillation():
    student_q_input = Input(shape=[],
                     dtype=tf.string,name='std_q')
    student_d_input = Input(shape=[],
                     dtype=tf.string,name='std_d')
    
    teacher_q_input = Input(shape=(512),name='tea_q')
    teacher_d_input = Input(shape=(512),name='tea_d')

    encoded_std_q = USE_student(student_q_input)
    encoded_std_d = USE_student(student_d_input) 

    mse_loss = tf.keras.losses.MeanSquaredError()

    mse_loss_q = mse_loss(teacher_q_input,encoded_std_q)*mse_factor_q
    mse_loss_d = mse_loss(teacher_d_input,encoded_std_d)*mse_factor_d
    mse_loss_qd = mse_loss(teacher_d_input,encoded_std_q)*mse_factor_qd
    mse_loss_all = (mse_loss_q+mse_loss_d+mse_loss_qd)*mse_factor

    distillation_network = Model(inputs=[student_q_input, student_d_input,teacher_q_input, teacher_d_input], outputs=mse_loss_all)

    distillation_network.add_loss(mse_loss_all)

    return distillation_network


def training_generator():
    idx = 0
    while True:
        student_q,_,student_d = all_data[idx]
        teacher_q = question_teacher_encoded[idx]
        teacher_d = doc_teacher_encoded[idx]
        idx += 1 
        yield student_q,teacher_q,student_d,teacher_d

corpus = arg_config.corpus
mode = arg_config.corpus_mode
languages = arg_config.languages.split('_')
top_start = arg_config.top_start
top_end = arg_config.top_end
context_mode = arg_config.context_mode
patient = 0
batch_size = arg_config.batch_size
num_epoch = arg_config.num_epoch
patient_limit = num_epoch*0.1
if patient_limit < 20:
    patient_limit = 20
use_mode = arg_config.use_mode
teacher = arg_config.teacher
learning_rate = float(arg_config.learning_rate)
if arg_config.dropout_rate != None:
    drop_out = 0
else:
    drop_out = arg_config.dropout_rate

mse_factor_q = float(arg_config.mse_factor_q)
mse_factor_d = float(arg_config.mse_factor_d)
mse_factor_qd = float(arg_config.mse_factor_qd)
mse_factor = arg_config.mse_factor

if use_mode == 'small':
    hub_load = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    USE_student =  hub.KerasLayer(hub_load,
                            input_shape=(),
                            output_shape = (512),
                            dtype=tf.string,
                            trainable=True)
    hub_load_2 = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    USE_teacher =  hub.KerasLayer(hub_load_2,
                            input_shape=(),
                            output_shape = (512),
                            dtype=tf.string,
                            trainable=False)
else:
    hub_load = hub.load(f"models/{corpus}/{use_mode}")
    USE_student =  hub.KerasLayer(hub_load,
                            input_shape=(),
                            output_shape = (512),
                            dtype=tf.string,
                            trainable=True)
    hub_load_2 = hub.load(f"models/{corpus}/{use_mode}")
    USE_teacher =  hub.KerasLayer(hub_load_2,
                            input_shape=(),
                            output_shape = (512),
                            dtype=tf.string,
                            trainable=False)
    use_mode = 'best_teacher'

hub_load_small = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
USE_small =  hub.KerasLayer(hub_load_small,
                        input_shape=(),
                        output_shape = (512),
                        dtype=tf.string,
                        trainable=False)


f_name = f"models/{corpus}/finetuned_USE_{corpus}_{mode}_en-{arg_config.languages}_top{top_start}-{top_end}_q-d-distillation_{mse_factor}MSE_{mse_factor_q}MSEq_{mse_factor_d}MSEd_{mse_factor_qd}MSEqd_{learning_rate}LR_teacher_{use_mode}_batchsize_{batch_size}_acc_metric_3term"

print("Setups:")
print(f"Corpus:{corpus} Mode:{mode}")
print(f"Language:{languages}")
print(f"top-start:{top_start} top-end:{top_end}")
print(f"Context:{context_mode} Patient Limit:{patient_limit}")
print(f"USE Model: {use_mode}")
print(f"Learning rate:{learning_rate} Replace:{arg_config.replace}")
print(f"MSE ALL:{mse_factor} Q:{mse_factor_q} D:{mse_factor_d} QD:{mse_factor_qd}")
print(f"file:{f_name}")
print()

if arg_config.replace == 'False' and os.path.isdir(f_name):
    raise Exception('File already have')

all_data = []
dev_data = []
teacher_all = []
doc_all = []
df_question={}
df_doc = {}
for lan in languages: 
    
    file_name = f'distillation_en-{lan}.csv'
    df_merged = pd.read_csv(f'data_preprocess/{corpus}/{mode}/distillation/{file_name}')
    if arg_config.shuffle == 'True':
        df_merged = df_merged.sample(frac=1, random_state=42)
    all_data += df_merged.values.tolist()
    teacher_all += df_merged.teacher.to_list()
    doc_all += df_merged.doc.to_list()

    doc_temp = pd.read_csv(f'data_preprocess/{corpus}/dev/{corpus.lower()}_doc_en-{lan}.csv')
    question_temp = pd.read_csv(f'data_preprocess/{corpus}/dev/{corpus.lower()}_question_en-{lan}.csv')
    df_question.update({
        f'en-{lan}':question_temp
    })  
    df_doc.update({
        f'en-{lan}':doc_temp
    })  




df_dev = pd.DataFrame(dev_data, columns =['student', 'teacher', 'doc'])

question_teacher_encoded = eval_fnc.batch_encode(USE_teacher,np.array(teacher_all),bs=120)
doc_teacher_encoded = eval_fnc.batch_encode(USE_teacher,np.array(doc_all),bs=120)

question_teacher_dev_encoded = eval_fnc.batch_encode(USE_teacher,df_dev['teacher'].to_numpy(),bs=120)
doc_teacher_dev_encoded = eval_fnc.batch_encode(USE_teacher,df_dev['doc'].to_numpy(),bs=120)

question_student_dev = df_dev['student'].to_list()
doc_student_dev = df_dev['doc'].to_list()

if __name__ == "__main__": 
    net = USE_distillation() 
    net.compile(optimizer=Adam(learning_rate=learning_rate))
    
    num_steps = int(len(all_data)/batch_size)
    previous_loss = math.inf
    check = 0
    dev_loss_save = []
    train_loss_save = []
    previous_score = -math.inf
    eval_step = 0
    for e in range(num_epoch):
        print("-" * 50)
        print(f"Epoch {e}")
        losses = []
        eval_count = 0
        training_gen = training_generator()
        check = e
        if check == 1:
            print('Check teacher pre and after')
            new_encoded = USE_teacher(np.array(teacher_all)).numpy()
            if (question_teacher_encoded != new_encoded).all():
                print(f"teacher pre:{pre_encoded[0]}")
                print(f"student after:{new_encoded[0]}")
                raise Exception('Teacher Changed')
            else:
                del teacher_all
                del doc_all
                print('Pass!')

        for i in tqdm(range(num_steps)):
            batch_student_q = []
            batch_student_d = []
            batch_teacher_d = []
            batch_teacher_q = []
            for j in range(batch_size):
                student_q_, teacher_q_, student_d_, teacher_d_ = next(training_gen)
                batch_student_q.append([student_q_])
                batch_student_d.append([student_d_])
                batch_teacher_d.append(teacher_d_)
                batch_teacher_q.append(teacher_q_)

            output = net.train_on_batch([np.array(batch_student_q),np.array(batch_student_d),
                                         np.array(batch_teacher_q),np.array(batch_teacher_d)],return_dict=True)
            losses.append(output['loss'])

#             if i%eval_step == 0 and eval_count < 4:
            if eval_step%100 == 0:
                
                eval_count+=1
                p_at_1 = 0
                p_at_5 = 0
                p_at_10 = 0
                mrr_at_10 = 0
                
                for lan in languages: 
                    question_id = df_question[f'en-{lan}']['doc_id'].to_list()
                    doc_context_id = df_doc[f'en-{lan}']['doc_id'].to_list()
                    doc_context_encoded = eval_fnc.batch_encode(USE_student,np.array(df_doc[f'en-{lan}']['doc'].to_list()))
                    questions = eval_fnc.batch_encode(USE_student,np.array(df_question[f'en-{lan}']['question'].to_list()))
                    top_1,top_5,top_10,mrr = eval_fnc.evaluate_inner(question_id,questions,doc_context_id,doc_context_encoded)
                        
                    precision1 = top_1 / len(questions)
                    precision5 = top_5 / len(questions)
                    precision10 = top_10 / len(questions)
                    p_at_1+=precision1
                    p_at_5+=precision5
                    p_at_10+=precision10
                    mrr_at_10+=mrr
                    
                    question_id = df_question[f'en-{lan}']['doc_id'].to_list()
                    doc_context_id = df_doc[f'en-{lan}']['doc_id'].to_list()
                    doc_context_encoded = eval_fnc.batch_encode(USE_teacher,np.array(df_doc[f'en-{lan}']['doc'].to_list()))
                    questions = eval_fnc.batch_encode(USE_teacher,np.array(df_question[f'en-{lan}']['question'].to_list()))
                    top_1,top_5,top_10,mrr = eval_fnc.evaluate_inner(question_id,questions,doc_context_id,doc_context_encoded)
                        
                    precision1 = top_1 / len(questions)
                    precision5 = top_5 / len(questions)
                    precision10 = top_10 / len(questions)
                    

                p_at_1/=len(languages)
                p_at_5/=len(languages)
                p_at_10/=len(languages)
                mrr_at_10/=len(languages)
                print(f"Epoch {e} Training step: {i} Train loss = {sum(losses)/len(losses)} MRR@10: {mrr_at_10}")
                print(f"Top1:{p_at_1:.3f} Top5:{p_at_5:.3f} Top10:{p_at_10:.3f} avg:{(p_at_1+p_at_5+p_at_10)/3:.3f}")
                
                
                if p_at_1 > previous_score:
                    print('Saved Model.......')
                    tf.saved_model.save(hub_load, f_name)
                    previous_score = p_at_1
                    patient = 0
                else:
                    if (patient%10)==0:
                        learning_rate = learning_rate*0.5
                    K.set_value(net.optimizer.learning_rate, learning_rate)  # set new learning_rate
                    patient+=1
                    print(f'Learning rate decreased from {learning_rate/0.5} to {learning_rate}')
                if patient >= patient_limit:
                    if e > 1:
                        raise Exception(f"Reach the patient value: Training finished")
                    pass
            eval_step+=1
