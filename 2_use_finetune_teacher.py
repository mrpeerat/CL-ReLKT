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

iimport tensorflow_hub as hub
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

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        if arg_config.loss_mode == 'cos':
            p_dist = 1-tf.linalg.diag_part(K.dot(anchor,tf.transpose(positive)))
            n_dist = 1-tf.linalg.diag_part(K.dot(anchor,tf.transpose(negative)))
            return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
        else:
            p_dist = K.sum(K.square(anchor-positive), axis=-1)
            n_dist = K.sum(K.square(anchor-negative), axis=-1)
            return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def USE_TripletNetwork(margin):
    anchor = Input(shape=[],
                   dtype=tf.string)
    
    positive = Input(shape=[],
                     dtype=tf.string)
    
    negative = Input(shape=[],
                     dtype=tf.string)
    
    encoded_a = USE(anchor)
    encoded_p = USE(positive)
    encoded_n = USE(negative)
    
    loss_layer = TripletLossLayer(alpha=margin,
                                  name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    
    triplet_network = Model(inputs=[anchor, positive, negative], outputs=loss_layer)
    return triplet_network

def sim_search(question_encoded,para_encoded):
    query_map = np.full(para_encoded.shape, question_encoded)
    sim_score = np.array([*map(np.inner,query_map,para_encoded)])
    return np.argsort(1-sim_score)[:],1-sim_score

def batch_encode(context,bs = 120):
    batch_encoded = []
    for i in range(len(context)//bs+1):
        batch_encoded.append(USE(context[(i*bs):((i+1)*bs)]).numpy()) 
    batch_encoded = np.concatenate(batch_encoded,0)
    return np.array(batch_encoded)

def negative_mining(triplet_data):
    sample_size = arg_config.hard+arg_config.semi_hard
    doc_id,question,para = triplet_data
    question_encoded = USE(np.array([question])).numpy()
    positive_encoded = USE(np.array([para])).numpy()
    d_a_p = 1-np.inner(question_encoded,positive_encoded)[0][0]

    para_encoded = np.array(df_doc_train[df_doc_train['doc_id']!=doc_id].doc_encoded.to_list())
    para_edited = np.array(df_doc_train[df_doc_train['doc_id']!=doc_id].doc.to_list())
    
    idx_sorted,s_score = sim_search(question_encoded,para_encoded)

    a_p_n_data = []
    hard = 0; s_hard = 0
    error = 0
    for idx in idx_sorted:
        d_a_n = s_score[idx]
        if len(a_p_n_data) < sample_size:
            if d_a_n < d_a_p and hard < arg_config.hard: # hard negative
                a_p_n_data.append([question,para,para_edited[idx]])
                hard+=1
            elif d_a_p < d_a_n and d_a_n < (d_a_p+arg_config.margin) and s_hard < arg_config.semi_hard:
                a_p_n_data.append([question,para,para_edited[idx]])
                s_hard+=1
    if hard+s_hard == 0:
        error = 1

    return a_p_n_data,error


def training_generator():
    idx = 0
    while True:
        triplets = training_data[idx]
        idx += 1
        yield triplets[0],triplets[1],triplets[2]

def update_negative_mining():
    all_data_function = [] 
    for _,item in enumerate(all_data[:]):     
        data_temp,error = negative_mining(item)
        if error == 0:
            all_data_function += data_temp
    all_data_df = pd.DataFrame(all_data_function,columns =['anchor', 'pos','neg'])
    all_data_df = all_data_df.drop_duplicates()
    all_data_df = all_data_df.sample(frac=1)
    all_data_function = all_data_df.values.tolist()
    return all_data_function


def update_both_negative_mining(): #update only hard negative sample
    all_data_function = [] 
    for _,item in enumerate(training_data[:]):     
        anchor_train, positive_train, negative_train = item
        anchor_encoded = USE(np.array([anchor_train])).numpy()
        positive_encoded = USE(np.array([positive_train])).numpy()
        negative_encoded = USE(np.array([negative_train])).numpy()
        d_a_p = 1-np.inner(anchor_encoded,positive_encoded)
        d_a_n = 1-np.inner(anchor_encoded,negative_encoded)

        if d_a_n < d_a_p and arg_config.hard_update == 'True': # hard
            all_data_function.append([anchor_train, positive_train, negative_train])
        elif d_a_p < d_a_n and d_a_n < (d_a_p+arg_config.margin) and arg_config.semi_hard_update == 'True': #semi-hard
            all_data_function.append([anchor_train, positive_train, negative_train])
    return all_data_function


corpus = arg_config.corpus
mode = arg_config.corpus_mode
languages = arg_config.languages.split('_')
top_start = arg_config.top_start
top_end = arg_config.top_end
context_mode = arg_config.context_mode
patient = 0
batch_size = arg_config.batch_size
num_epoch = arg_config.num_epoch
patient_limit = num_epoch*0.2
global_index_batch = 0
use_mode = arg_config.use_mode
training_data = []
training_data_temp = []
f_name = f"models/{corpus}/finetuned_USE_{corpus}_{mode}_en-{arg_config.languages}_top{top_start}-{top_end}_{context_mode}_{use_mode}_{arg_config.margin}_{arg_config.batch_update}BatchUpdate_{arg_config.hard}hard_{arg_config.semi_hard}shard_{arg_config.loss_mode}_HardUpdate{arg_config.hard_update}_SemiHardUpdate{arg_config.semi_hard_update}_RegFalse"

print("Setups:")
print(f"Corpus:{corpus} Mode:{mode}")
print(f"Language:{languages}")
print(f"top-start:{top_start} top-end:{top_end}")
print(f"Context:{context_mode} Patient Limit:{patient_limit}")
print(f"USE Mode: {use_mode} Margin: {arg_config.margin}")
print(f"Hard:{arg_config.hard} Semi-hard:{arg_config.semi_hard}")
print(f"Hard update:{arg_config.hard_update} Semi-hard update:{arg_config.semi_hard_update}")
print(f"LossMode:{arg_config.loss_mode}")
print(f"name:{f_name}")
print()


if arg_config.replace == 'False' and os.path.isdir(f_name):
    raise Exception('File already have')

all_data = []
df_question={}
for lan in languages: 
    df_doc_train = pd.read_csv(f'data_preprocess/{corpus}/train/{corpus.lower()}_doc_en-{lan}.csv')

    df_doc = pd.read_csv(f'data_preprocess/{corpus}/dev/{corpus.lower()}_doc_en-{lan}.csv')
    question_temp = pd.read_csv(f'data_preprocess/{corpus}/dev/{corpus.lower()}_question_en-{lan}.csv')
    df_question.update({
        f'en-{lan}':question_temp
    })  

    file_name = f'triplet_en-{lan}_top{top_start}-{top_end}_{context_mode}.csv'
    df = pd.read_csv(f'data_preprocess/{corpus}/{mode}/triplet/{file_name}')
    df = df.sample(frac=1)
    df = df.dropna()
    all_data += df.values.tolist()


if __name__ == "__main__":

    if use_mode == 'small':
        hub_load = hub.load(f"https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    elif use_mode == 'large':
        hub_load = hub.load(f"https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    else:
        hub_load = hub.load(f"models/{corpus}/best_model/{use_mode}")
    USE =  hub.KerasLayer(hub_load,
                           input_shape=(),
                           output_shape = (512),
                           dtype=tf.string,
                           trainable=True)
    net = USE_TripletNetwork(arg_config.margin) 
    learning_rate = 1e-5
    net.compile(optimizer=Adam(learning_rate=learning_rate),loss=TripletLossLayer.triplet_loss)
    previous_score = -math.inf
    

    for e in range(num_epoch):
        print("-" * 50)
        print(f"Epoch {e}")
        losses = []
        
        if e%arg_config.batch_update == 0: # update all the dataset
            print('Updating data....')
            print('Doc encoding.....')
            doc_encoded_list = eval_fnc.batch_encode(USE,np.array(df_doc_train['doc'].to_list()))
            try:
                df_doc_train = df_doc_train.drop(columns=['doc_encoded'])
            except:
                pass
            df_doc_train['doc_encoded'] = doc_encoded_list.tolist()
            print('Data Updating......')
            training_data = update_negative_mining()
            print(f'Data updated from {len(all_data)} to {np.array(training_data).shape}......')
        else: #update only hard negative sample
            if arg_config.hard_update == 'True' or arg_config.semi_hard_update == 'True':
                print('Update hard and semi hard samples')
                print(f'Data updated from {len(training_data)}')
                doc_encoded_list = eval_fnc.batch_encode(USE,np.array(df_doc_train['doc'].to_list()))
                try:
                    df_doc_train = df_doc_train.drop(columns=['doc_encoded'])
                except:
                    pass
                df_doc_train['doc_encoded'] = doc_encoded_list.tolist()
                training_data = update_both_negative_mining()
                print(f'To {np.array(training_data).shape}')

        training_gen = training_generator()
        num_steps = len(training_data)//batch_size
        eval_step = num_steps//4
        eval_count = 0
        for i in tqdm(range(num_steps)):
            batch_a = []
            batch_p = []
            batch_n = []        
            for j in range(batch_size):
                anchor_b, pos_b, neg_b = next(training_gen)
                batch_a.append([anchor_b])
                batch_p.append([pos_b])
                batch_n.append([neg_b])

            output = net.train_on_batch([np.array(batch_a),
                                np.array(batch_p), 
                                np.array(batch_n)], return_dict=True)
            losses.append(output['loss'])
 
            if i%eval_step == 0 and eval_count < 4:
                eval_count+=1
                p_at_1 = 0
                p_at_5 = 0
                p_at_10 = 0
                mrr_at_10 = 0
                doc_context_id = df_doc['doc_id'].to_list()
                for lan in languages: 
                    question_id = df_question[f'en-{lan}']['doc_id'].to_list()
                    doc_context_encoded = eval_fnc.batch_encode(USE,np.array(df_doc['doc'].to_list()))
                    questions = eval_fnc.batch_encode(USE,np.array(df_question[f'en-{lan}']['question'].to_list()))
                    top_1,top_5,top_10,mrr = eval_fnc.evaluate_dot_normal(question_id,questions,doc_context_id,doc_context_encoded)
                    
                    precision1 = top_1 / len(questions)
                    precision5 = top_5 / len(questions)
                    precision10 = top_10 / len(questions)
                    p_at_1+=precision1
                    p_at_5+=precision5
                    p_at_10+=precision10
                    mrr_at_10+=mrr

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
                    learning_rate = learning_rate*0.5
                    K.set_value(net.optimizer.learning_rate, learning_rate)  # set new learning_rate
                    patient+=1
                    print(f'Learning rate decreased from {learning_rate/0.5} to {learning_rate}')
                if patient >= patient_limit:
                    raise Exception(f"Reach the patient value: Training finished")
