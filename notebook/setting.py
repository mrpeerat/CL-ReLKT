import tqdm
import pandas as pd
import numpy as np
# import MicroTokenizer
try:
    from rank_bm25 import BM25Okapi, BM25Plus
except:
    pass
# from pythainlp.tokenize import word_tokenize 
from glob import glob
import pickle
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, models

def plot_fnc(losses,mode):
    steps = range(len(losses))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Steps*100',fontsize=30)
    ax.set_ylabel('Loss',fontsize=30)
    ax.plot(steps,losses,c="r", label=f'{mode}-loss')
    plt.show()
def plot_function(path):
    with open(f'{path}/dev_loss.pickle', 'rb') as f:
        dev_loss = pickle.load(f)
    with open(f'{path}/train_loss.pickle', 'rb') as f:
        training_loss = pickle.load(f)   
    training_loss_edited = [sum(training_loss[idx-3:idx+1])/4 for idx,item in enumerate(training_loss) if (idx+1)%4 == 0]
    plot_fnc(dev_loss,mode='Dev')
    plot_fnc(training_loss_edited,mode='Train')

# +
def sim_search(question_encoded,doc_encoded):
    query_map = np.full(doc_encoded.shape, question_encoded)
    sim_score = np.array([*map(np.inner,query_map,doc_encoded)])
    return np.argsort(sim_score)[::-1]

def evaluate(question_id,question_all,context_id,context_all,mrr_rank=10,status=True):
    top_1 = 0; top_5 = 0; top_10 = 0;
    mrr_score = 0
    context_id = np.array(context_id)
    sim_score = np.inner(question_all,context_all)
    if status == True:
        status_bar = enumerate(tqdm.tqdm(sim_score))
    else:
        status_bar = enumerate(sim_score)
    for idx,sim in status_bar:
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


# -

def evaluate_para(question_id_,question_all,context_id,context_all,mrr_rank=10):
    top_1 = 0; top_5 = 0; top_10 = 0;
    mrr_score = 0
    context_id = np.array(context_id)
    for idx,question in enumerate(tqdm.tqdm(question_all)):
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


def bm25_encode(contexts,language):
    if language == 'zh':
        context_split = [*map(MicroTokenizer.cut,contexts)]
    elif language == 'th':
        context_split = [word_tokenize(x, engine="newmm") for x in contexts]
    else:
        context_split = [x.split(' ') for x in contexts]
        
    context_encoded = BM25Okapi(context_split)
    return context_encoded

def bm25_scoring(question,context_encoded,language):
    if language == 'zh':
        tokenized_query = MicroTokenizer.cut(question)
    elif language == 'th':
        tokenized_query = word_tokenize(question, engine="newmm")
    else:
        tokenized_query = question.split(' ')
        
    ranking = context_encoded.get_scores(tokenized_query)
    return np.argsort(-ranking)

def evaluate_bm25(question_id,question_all,context_id,context_all,lan,status=True):
    top_1 = 0; top_5 = 0; top_10 = 0;
    index_answer = []
    if status == True:
        status_bar = enumerate(tqdm.tqdm(question_all))
    else:
        status_bar = enumerate(question_all)
        
    for idx,question in status_bar:
        index = bm25_scoring(question,context_all,lan)
        index = list(index)
        index_answer.append(index)
        if context_id[index[0]] == question_id[idx]:
            top_1+=1
        for i in range(0,5):
            if context_id[index[i]] == question_id[idx]:
                top_5+=1
                break
        for i in range(0,10):
            if context_id[index[i]] == question_id[idx]:
                top_10+=1
                break
    return top_1,top_5,top_10

# +
def batch_encode_fc(encoder,context,fc_layer,bs=10):
    batch_encoded = []
    for i in range(len(context)//bs+1):
        batch_encoded.append(fc_layer(encoder(context[(i*bs):((i+1)*bs)])).numpy()) 
    batch_encoded = np.concatenate(batch_encoded,0)
    return np.array(batch_encoded)

def batch_encode(encoder,context,bs = 10):
    batch_encoded = []
    for i in range(len(context)//bs+1):
        batch_encoded.append(encoder(context[(i*bs):((i+1)*bs)]).numpy()) 
 
    batch_encoded = np.concatenate(batch_encoded,0)
    return np.array(batch_encoded)


# -

def sent_bert_encode(sent_bert,mode,lan,df_con,df_que):
    doc_context_id = df_con['doc_id'].to_list()
    doc_context_encoded = sent_bert.encode(df_con['doc'].to_list())

    question_id = df_que[f'en-{lan}']['doc_id'].to_list()
    questions = sent_bert.encode(df_que[f'en-{lan}']['question'].to_list())

    top_1,top_5,top_10,mrr = evaluate(question_id,questions,doc_context_id,doc_context_encoded)

    print(f'SB-{mode}-{lan}')
    precision = top_1 / len(questions)
    print(f"Traninng Score P@1: {precision:.3f}")
    precision = top_5 / len(questions)
    print(f"Traninng Score P@5: {precision:.3f}")
    precision = top_10 / len(questions)
    print(f"Traninng Score P@10: {precision:.3f}")
    print(f"Mrr score:{mrr:.3f}")

def load_weight(path,share_encoded,q_or_d):
    w = np.load(f"../models/{path}/{q_or_d}_weight.npy")
    b = np.load(f"../models/{path}/{q_or_d}_bias.npy")
    share_encoded.set_weights([w,b])
    return share_encoded

def load_weight_before(path,share_encoded,q_or_d):
    w = np.load(f"../models/{path}/{q_or_d}_weight_before.npy")
    b = np.load(f"../models/{path}/{q_or_d}_bias_before.npy")
    share_encoded.set_weights([w,b])
    return share_encoded

def embedding_model(model_name, pooling_mean=True, pooling_cls=False, pooling_max=False):
  word_embedding_model = models.Transformer(model_name)
  pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=pooling_mean,
                               pooling_mode_cls_token=pooling_cls,
                               pooling_mode_max_tokens=pooling_max)

  model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
  return model

def sentence_encode(model, input_text):
  sentence_embedding = model.encode(input_text)
  return sentence_embedding
