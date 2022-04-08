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

def USE_TripletNetwork(margin=0.2):
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


def training_generator():
    idx = 0
    while True:
        triplets = all_data[idx]
        idx += 1
        yield triplets


def training_generator_random():
    idx = 0
    while True:
        triplets = all_data[idx]
        rand_idx = idx
        while rand_idx == idx:
            rand_idx = np.random.randint(len(all_data), size=1)[0]
        neg_random = all_data[rand_idx][2]
        idx += 1
        yield triplets[0],triplets[1],neg_random


corpus = arg_config.corpus
mode = arg_config.corpus_mode
languages = arg_config.languages.split('_')
top_start = arg_config.top_start
top_end = arg_config.top_end
context_mode = arg_config.context_mode
patient = 0
batch_size = arg_config.batch_size
use_mode = arg_config.use_mode
num_epoch = arg_config.num_epoch
patient_limit = num_epoch*0.25
margin_trip = arg_config.margin
learning_rate = float(arg_config.learning_rate)

f_name = f"models/{corpus}/finetuned_USE_{corpus}_{mode}_en-{arg_config.languages}_top{top_start}-{top_end}_{context_mode}_{margin_trip}margin_{arg_config.warup_steps}WarmStep_{arg_config.loss_mode}Loss_{learning_rate}LR_RegFalse_FCFalse_new"

if arg_config.replace == 'False' and os.path.isdir(f_name):
    raise Exception('File already have')

print("Setups:")
print(f"Corpus:{corpus} Mode:{mode}")
print(f"Language:{languages}")
print(f"top-start:{top_start} top-end:{top_end}")
print(f"Context:{context_mode} Patient Limit:{patient_limit}")
print(f"Margin:{margin_trip} Warmup Step:{arg_config.warup_steps}")
print(f"Loss distance:{arg_config.loss_mode}")
print(f"name:{f_name}")
print()
# -

df_question = {}
all_data = []
for lan in languages: 
    df_doc = pd.read_csv(f'data_preprocess/{corpus}/dev/{corpus.lower()}_doc_en-{lan}.csv')
    question_temp = pd.read_csv(f'data_preprocess/{corpus}/dev/{corpus.lower()}_question_en-{lan}.csv')
    df_question.update({
        f'en-{lan}':question_temp
    })  

    file_name = f'triplet_en-{lan}_top{top_start}-{top_end}_{context_mode}_new.csv'
    df = pd.read_csv(f'data_preprocess/{corpus}/{mode}/triplet/{file_name}')
    df = df.sample(frac=1)
    df = df.dropna()
    all_data += df.values.tolist()

if __name__ == "__main__":
    if use_mode == 'small':
        hub_load = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        USE =  hub.KerasLayer(hub_load,
                            input_shape=(),
                            output_shape = (512),
                            dtype=tf.string,
                            trainable=True)
    else:
        hub_load = hub.load(f"models/{corpus}/best_model/{use_mode}")
        USE =  hub.KerasLayer(hub_load,
                            input_shape=(),
                            output_shape = (512),
                            dtype=tf.string,
                            trainable=True)
    
    net = USE_TripletNetwork(margin_trip) 
    net.compile(optimizer=Adam(learning_rate=learning_rate),loss=TripletLossLayer.triplet_loss)
    
    num_steps = int(len(all_data)/batch_size)
    previous_score = -math.inf
    eval_step = 0
    for e in range(num_epoch):
        print("-" * 50)
        print(f"Epoch {e}")
        losses = []
        training_gen = training_generator()
        training_gen_random = training_generator_random()
        eval_count=0
        for i in tqdm(range(num_steps)):
            batch_a = []
            batch_p = []
            batch_n = []
            for j in range(batch_size):
                if e < arg_config.warup_steps:
                    anchor, pos, neg = next(training_gen)
                else:
                    if lan != 'all':
                        anchor, pos, neg = next(training_gen_random)
                    else:
                        anchor, pos, neg = next(training_gen)
                batch_a.append([anchor])
                batch_p.append([pos])
                batch_n.append([neg])

            output = net.train_on_batch([np.array(batch_a),
                                np.array(batch_p), 
                                np.array(batch_n)], return_dict=True)
            losses.append(output['loss'])
  
            if eval_step%100 == 0:
                eval_count+=1
                p_at_1 = 0
                p_at_5 = 0
                p_at_10 = 0
                mrr_at_10 = 0
                doc_context_id = df_doc['doc_id'].to_list()
                for lan in languages: 
                    question_id = df_question[f'en-{lan}']['doc_id'].to_list()
                    
                    if arg_config.loss_mode == 'cos':
                        doc_context_encoded = eval_fnc.batch_encode(USE,np.array(df_doc['doc'].to_list()))
                        questions = eval_fnc.batch_encode(USE,np.array(df_question[f'en-{lan}']['question'].to_list()))
                        top_1,top_5,top_10,mrr = eval_fnc.evaluate_dot_normal(question_id,questions,doc_context_id,doc_context_encoded)
                    else:
                        doc_context_encoded = eval_fnc.batch_encode(USE,np.array(df_doc['doc'].to_list()))
                        questions = eval_fnc.batch_encode(USE,np.array(df_question[f'en-{lan}']['question'].to_list()))
                        top_1,top_5,top_10,mrr = eval_fnc.evaluate(question_id,questions,doc_context_id,doc_context_encoded)
                        
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
                    if (patient%5)==0:
                        learning_rate = learning_rate*0.5
                    K.set_value(net.optimizer.learning_rate, learning_rate)  # set new learning_rate
                    patient+=1
                    print(f'Learning rate decreased from {learning_rate/0.5} to {learning_rate}')
                if patient >= patient_limit:
                    if e > 1:
                        raise Exception(f"Reach the patient value: Training finished")
                    pass
            eval_step+=1
