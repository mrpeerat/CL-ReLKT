import copy
import re
import os, sys
import json
import glob
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import Counter, defaultdict
from pprint import pprint

import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import heapq
import jsonlines
import numpy as np
from pythainlp.tokenize import word_tokenize as th_word_tokenize
from nltk.tokenize import word_tokenize as en_word_tokenize, sent_tokenize as en_sent_tokenize


def segment_sentence(text:str) -> List[str]:
    return en_sent_tokenize(text)

def find_sentences_matched_answer(answer: str, sentences: List[str]) -> List[Tuple[str, Tuple[int, int]]]:
    sentence_candidates = []
    for i, sentence in enumerate(sentences):
        if answer in sentence:
            search_obj = re.search(answer, sentence)
            sentence_candidates.append((sentence, answer, search_obj.span(0)))

    return sentence_candidates

def find_gt_sentence(qa_item, document):
    
    segments = segment_sentence(document)
    pass

def compute_dot_product_sim(query, keys):
    scores = np.inner(query, keys)
    return scores

def select_candidate(question, sentence_candidates, method='first', model=None, topk=10):
    if method == 'first':
        return sentence_candidates[0]
    elif method == 'vsim_dot':
        q_vector = model(question)
        s_vectors = model(sentence_candidates)
        scores = compute_dot_product_sim(q_vector, s_vectors)

        K = min(topk, len(sentence_candidates))

        topk_indices = heapq.nlargest(K, range(len(sentence_candidates)), scores.take)


        return [(sentence_candidates[i], float(scores[0][i])) for i in topk_indices]
    else:
        raise NotImplementedError()

def mine_prompt_gt(args):
    context, question, answer, answer_start = args
    sentences = segment_sentence(context)
    

    acc_len = 0
    selected_sentence = '<NA>'
    for i, sentence_candidate in enumerate(sentences):
        if answer_start >= acc_len and answer_start <=acc_len + len(sentence_candidate):
            selected_sentence = sentence_candidate
            
        acc_len+=len(sentence_candidate)

    prompt_template = 'Question: {} Answer: {}'
    
    prompt = prompt_template.format(question.strip(), selected_sentence)

    return selected_sentence

def load_model(directory: str):
    model = hub.load(directory)
    return model

def evaluate(dataset, key_name, topk=10):
    matches_at1 = 0
    accuracy = None
    matches_at_k = Counter()
    for item in dataset:
        if item[key_name][0][0] == item['gt_sentence']:
            matches_at1+=1
        for k in range(min(topk, len(item[key_name]))):
            if item[key_name][k][0] == item['gt_sentence']:
                matches_at_k[k+1] +=1
                break
    # key starts from 1 to 10
    for k in range(2, topk+1):      
       
        matches_at_k[k] += matches_at_k[k-1]
    precision_at_k = {}
    for k, count in matches_at_k.items():        
        precision_at_k[k] = float(count / len(dataset))
        
    accuracy = float(matches_at1 / len(dataset))

    return accuracy, precision_at_k

def run_online_prompt_mining(dataset, prefix, model):
    result_dataset = copy.deepcopy(dataset)
#     print(f'prefix: {prefix}')
    for item in tqdm(result_dataset, total=len(result_dataset)):
        question = item['question']
        segmented_context = item['segmented_context']
        selected_candidate = select_candidate(question=question, sentence_candidates=segmented_context,
                         method='vsim_dot', model=model)
        item[f'{prefix}@top_sentence'] = selected_candidate
    print('\n\tEvaluation result:')
        
    key_name = f'{prefix}@top_sentence'
    accuracy, precision_at_k = evaluate(result_dataset, key_name)
    print(f'\t - Accuracy: {accuracy:.4f}')
    print(f'\t - precision_at_k:')
    pprint(precision_at_k)
    evaluation_result = (accuracy, precision_at_k)
    return evaluation_result, result_dataset