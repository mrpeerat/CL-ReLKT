from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import evaluate as eval_fnc
import tensorflow as tf
from tensorflow.keras import backend as K

def batch_encode(encoder,context,bs = 150,fully_connec=False):
    batch_encoded = []
    for i in range(len(context)//bs+1):
        if fully_connec == False:
            batch_encoded.append(encoder(context[(i*bs):((i+1)*bs)]).numpy()) 
        else:
            raise Exception
    batch_encoded = np.concatenate(batch_encoded,0)
    return np.array(batch_encoded)

def batch_encode_fc(encoder,context,fc_layer,bs=10):
    batch_encoded = []
    for i in range(len(context)//bs+1):
        encdoed_context = encoder(context[(i*bs):((i+1)*bs)])
        batch_encoded.append(fc_layer(encdoed_context))
    batch_encoded_ = np.concatenate(batch_encoded,0)
    return tf.convert_to_tensor(batch_encoded_)


def sim_search(question_encoded,doc_encoded):
    query_map = np.full(doc_encoded.shape, question_encoded)
    sim_score = np.array([*map(np.inner,query_map,doc_encoded)])
    return np.argsort(sim_score)[::-1]

def evaluate(question_id,question_all,context_id,context_all,mrr_rank=10):
    top_1 = 0; top_5 = 0; top_10 = 0;
    mrr_score = 0
    context_id = np.array(context_id)
    for idx,question in enumerate(question_all):
        index_ = sim_search(question,context_all)
        index_edit = [context_id[x] for x in index_]
        try:
            idx_search = list(index_edit).index(question_id[idx])
        except:
            idx_search = 999999
        if idx_search == 0:
            top_1+=1
            top_5+=1
            top_10+=1
        elif idx_search < 5:
            top_5+=1
            top_10+=1
        elif idx_search < 10:
            top_10+=1  
        if idx_search < mrr_rank:
            mrr_score += (1/(idx_search+1))
    mrr_score/=len(question_all)
    return top_1,top_5,top_10,mrr_score

def evaluate_dot(question_id,question_all,context_id,context_all,mrr_rank=10):
    top_1 = 0; top_5 = 0; top_10 = 0;
    mrr_score = 0
    context_id = np.array(context_id)
    sim_score = 1-K.dot(question_all,tf.transpose(context_all)).numpy()
    for idx,sim in enumerate(sim_score):
        index = np.argsort(sim)
        print(sim)
        print('*'*50)
        print(index)
        raise Exception
        index_edit = [context_id[x] for x in index]
        idx_search = list(index_edit).index(question_id[idx])
        if idx_search == 0:
            top_1+=1
            top_5+=1
            top_10+=1
        elif idx_search < 5:
            top_5+=1
            top_10+=1
        elif idx_search < 10:
            top_10+=1  
        if idx_search < mrr_rank:
            mrr_score += (1/(idx_search+1))
    mrr_score/=len(question_all)
    return top_1,top_5,top_10,mrr_score

def evaluate_dot_normal(question_id,question_all,context_id,context_all,mrr_rank=10):
    top_1 = 0; top_5 = 0; top_10 = 0;
    mrr_score = 0
    context_id = np.array(context_id)
    sim_score = np.inner(question_all,context_all)
    
    for idx,sim in enumerate(sim_score):
        index = np.argsort(sim)[::-1]
        index_edit = [context_id[x] for x in index]
        try:
            idx_search = list(index_edit).index(question_id[idx])
        except:
            idx_search = 999999
        if idx_search == 0:
            top_1+=1
            top_5+=1
            top_10+=1
        elif idx_search < 5:
            top_5+=1
            top_10+=1
        elif idx_search < 10:
            top_10+=1  
        if idx_search < mrr_rank:
            mrr_score += (1/(idx_search+1))
    mrr_score/=len(question_all)
    return top_1,top_5,top_10,mrr_score


def evaluate_inner(question_id,question_all,context_id,context_all,mrr_rank=10):
    top_1 = 0; top_5 = 0; top_10 = 0;
    mrr_score = 0
    context_id = np.array(context_id)
    sim_score = np.inner(question_all,context_all)
    
    for idx,sim in enumerate(sim_score):
        index = np.argsort(sim)[::-1]
        index_edit = [context_id[x] for x in index]
        try:
            idx_search = list(index_edit).index(question_id[idx])
        except:
            idx_search = 999999
        if idx_search == 0:
            top_1+=1
            top_5+=1
            top_10+=1
        elif idx_search < 5:
            top_5+=1
            top_10+=1
        elif idx_search < 10:
            top_10+=1  
        if idx_search < mrr_rank:
            mrr_score += (1/(idx_search+1))
    mrr_score/=len(question_all)
    return top_1,top_5,top_10,mrr_score



def evaluate_para(question_id_,question_all,context_id,context_all,mrr_rank=10):
    top_1 = 0; top_5 = 0; top_10 = 0;
    mrr_score = 0
    context_id = np.array(context_id)
    for idx,question in enumerate(question_all):
        index = sim_search(question,context_all)
        index = list(index)
        if context_id[index[0]] == question_id_[idx]:
            top_1+=1
        for i in range(0,5):
            if context_id[index[i]] == question_id_[idx]:
                top_5+=1
                break
        for i in range(0,10):
            if context_id[index[i]] == question_id_[idx]:
                mrr_score += (1/(i+1))
                top_10+=1
                break
            
    mrr_score/=len(question_all)
    return top_1,top_5,top_10,mrr_score

