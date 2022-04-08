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
tf.config.experimental_run_functions_eagerly(True)

import tensorflow_hub as hub
import tensorflow_text
from tensorflow.keras.layers import Layer,Input,Dense,Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.activations import sigmoid, softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import math


corpus = arg_config.corpus
mode = arg_config.corpus_mode
# languages = ['en','ar','de','es','hi','vi','zh']
languages = arg_config.languages.split('_')
top_start = arg_config.top_start
top_end = arg_config.top_end
context_mode = arg_config.context_mode
patient = 0
batch_size = arg_config.batch_size
num_epoch = arg_config.num_epoch
patient_limit = num_epoch*0.1
margin_trip = arg_config.margin
warup_steps = arg_config.warup_steps

all_data = []
dev_data = []
for lan in languages: 
    file_name = f'triplet_en-{lan}_top{top_start}-{top_end}_{context_mode}.csv'
    df = pd.read_csv(f'data_preprocess/{corpus}/{mode}/triplet/{file_name}')
    df = df.dropna()
    all_data_lan = df.values.tolist()
    negative_data = df['negative'].to_list()
    all_data_lan, dev_data_lan = train_test_split(all_data_lan, test_size=0.1, random_state=42)
    all_data+=all_data_lan
    dev_data+=dev_data_lan

print("Setups:")
print(f"Corpus:{corpus} Mode:{mode}")
print(f"Language:{languages}")
print(f"top-start:{top_start} top-end:{top_end}")
print(f"Context:{context_mode} Patient Limit:{patient_limit}")
print(f"Margin:{margin_trip}")
print(f"Training data: {len(all_data)} Dev data: {len(dev_data)}")
print()

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
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
    
    encoded_a_f = share_encoded_que_f(encoded_a)
    encoded_a_f_norm = Lambda(lambda x: K.l2_normalize(x,axis=1))(encoded_a_f)
    
    encoded_p_f = share_encoded_doc_f(encoded_p)
    encoded_p_f_norm = Lambda(lambda x: K.l2_normalize(x,axis=1))(encoded_p_f)
    
    encoded_n_f = share_encoded_doc_f(encoded_n)
    encoded_n_f_norm = Lambda(lambda x: K.l2_normalize(x,axis=1))(encoded_n_f)
    
    loss_layer = TripletLossLayer(alpha=margin,
                                  name='triplet_loss_layer')([encoded_a_f_norm,encoded_p_f_norm,encoded_n_f_norm])
    
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


if __name__ == '__main__':
    a,p,n = zip(*dev_data)

    share_encoded_doc_f = Dense(512,name='d_shared')
    share_encoded_que_f = Dense(512,name='q_shared')
    hub_load = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    USE =  hub.KerasLayer(hub_load,
                           input_shape=(),
                           output_shape = (512),
                           dtype=tf.string,
                           trainable=True)
    net = USE_TripletNetwork(margin_trip) 
    learning_rate = 1e-5
    net.compile(optimizer=Adam(learning_rate=learning_rate),loss=TripletLossLayer.triplet_loss)
    
    num_steps = int(len(all_data)/batch_size)
    previous_loss = math.inf
    
    for e in range(num_epoch):
        print("-" * 50)
        print(f"Epoch {e}")
        losses = []
        training_gen = training_generator()
        training_gen_random = training_generator_random()
        for i in tqdm(range(num_steps)):
            batch_a = []
            batch_p = []
            batch_n = []
            for j in range(batch_size):
                if e < warup_steps:
                    anchor, pos, neg = next(training_gen)
                else:
                    if lan != 'all':
                        anchor, pos, neg = next(training_gen_random)
                        # anchor, pos, neg = next(training_gen)
                    else:
                        anchor, pos, neg = next(training_gen)
                batch_a.append([anchor])
                batch_p.append([pos])
                batch_n.append([neg])

            output = net.train_on_batch([np.array(batch_a),
                                np.array(batch_p), 
                                np.array(batch_n)], return_dict=True)
            losses.append(output['loss'])
  
            if i%100 == 0 or i == (num_steps):
                dev_losses = []
                for j in range(int(len(dev_data)/batch_size)):
                    dev_loss_1 = net.test_on_batch([np.array(a[j:j+32]),np.array(p[j:j+32]),np.array(n[j:j+32])], return_dict=True)['loss']
                    dev_losses.append(dev_loss_1)
                dev_loss = sum(dev_losses)/len(dev_losses)
                print(f"Dev loss: {dev_loss}")
                if dev_loss < previous_loss:
                    print('Saved Model.......')
                    f_name = f"models/finetuned_USE_{corpus}_{mode}_en-{arg_config.languages}_top{top_start}-{top_end}_{context_mode}_{margin_trip}margin_cldr"
                    tf.saved_model.save(hub_load, f_name)
                    doc_w,doc_b = share_encoded_doc_f.get_weights()
                    que_w,que_b = share_encoded_que_f.get_weights()
                    np.save(f'{f_name}/doc_weight',doc_w)
                    np.save(f'{f_name}/doc_bias',doc_b)
                    np.save(f'{f_name}/que_weight',que_w)
                    np.save(f'{f_name}/que_bias',que_b)
                    previous_loss = dev_loss
                    patient = 0
                else:
                    learning_rate = learning_rate*0.98
                    K.set_value(net.optimizer.learning_rate, learning_rate)  # set new learning_rate
                    patient+=1
                    print(f"Reduce LR from:{learning_rate*1.2} to {learning_rate}")
                if patient >= patient_limit:
                    raise Exception(f"Reach the patient value: Training finished")
        
        loss = sum(losses)/num_steps # Train loss
        print(f"Epoch {e}: Train loss = {loss} Dev loss = {dev_loss}")
