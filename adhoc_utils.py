import pandas as pd
import math
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
from transformers import AutoModel, AutoTokenizer 
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import pipeline
# from transformers import logging

import spacy
nlp = spacy.load("en_core_web_sm") # pos tagging

from nltk.stem import WordNetLemmatizer
wn = WordNetLemmatizer()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mask = tokenizer.mask_token

import gensim.downloader
glove_vectors = gensim.downloader.load('glove-twitter-25')


def flatten_data(data, subdata):
    test = [[(x.lower().strip("(s)"), subdata[i][x]) for x in subdata[i]] for i in data if i in subdata] + [(i.lower().strip("(s)"), data[i]) for i in data if i not in subdata]
    return dict([i for sublist in test for i in sublist if type(sublist) == list] + [i for i in test if type(i) != list])


# Takes text containing "<MASK>" token and return topk results for masked token prediction.
# print_results dictates whether to print filled sentences during function execution
def get_mask(text, model='bert-base-uncased', topk=10, show=False, filter_pos=False, lemmatize=False):
    if not "<MASK>" in text:
        print("Text should contain \"<MASK>\" token")
        print(text)
        return False
    
    unmasker = pipeline('fill-mask', model=model)
    text = text.replace("<MASK>", unmasker.tokenizer.mask_token)
    unmasked = unmasker(text, topk=topk)
    top_tokens = [(token['token_str'].replace(" ", ""), token['score']) for token in unmasked]

    if filter_pos:
        top_tokens = [(token,score) for (token,score) in top_tokens if nlp(token)[0].pos_ in ['NOUN', 'VERB']]
        k=topk
        while(len(top_tokens) < topk):
            print("k=",k,"topk=",topk,"len(top_tokens)=",len(top_tokens))
            k+=20
            unmasked = unmasker(text, topk=k)
            top_tokens = [(token['token_str'].replace(" ", ""), token['score']) for token in unmasked]
            top_tokens = [(token,score) for (token,score) in top_tokens if nlp(token)[0].pos_ in ['NOUN', 'VERB']]
    if lemmatize:
        top_tokens = [(wn.lemmatize(token), score) for (token, score) in top_tokens]
    if show:
        print(unmasked[:min(5, topk)])

    
    return top_tokens[:topk]
   

# !!! Old method, kept for reference !!!
# Takes text containing "<MASK>" token and return topk results for masked token prediction.
# print_results dictates whether to print filled sentences during function execution
def get_mask_old(text, model_name='bert-base-uncased', topk=10, show=True, lemmatize=False):
    if model_name == "bert-base-uncased": 
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name, return_dict = True)
    elif model_name == "vinai/bertweet-base":
        model = BertTokenizer.from_pretrained("vinai/bertweet-base")
        tokenizer = BertForMaskedLM.from_pretrained("vinai/bertweet-base", return_dict = True)
    elif model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForMaskedLM.from_pretrained(model_name, return_dict = True)
    else:
        print("Unrecognized model")
        return False
    
    if "<MASK>" in text:
        text = text.replace("<MASK>", tokenizer.mask_token)
        input = tokenizer.encode_plus(text, return_tensors = "pt")
        mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
        softmax = F.softmax(model(**input).logits, dim = -1)
        mask_word = softmax[0, mask_index, :]
        top_tokens = torch.topk(mask_word, topk, dim = 1)[1][0]
        if show:
            for token in top_tokens[:5]:
                word = tokenizer.decode([token]).strip()
                new_sentence = text.replace(tokenizer.mask_token, word)
                print("{0:.5f}".format(mask_word[0][token].detach().numpy()[()]), new_sentence)
        if lemmatize:
            return [(wn.lemmatize(tokenizer.decode([token]).strip()), mask_word[0][token].detach().numpy()[()]) for token in top_tokens]
        else:
            return [(tokenizer.decode([token]).strip(), mask_word[0][token].detach().numpy()[()]) for token in top_tokens]
    else:
        print("Text should contain \"<MASK>\" token")
        print(text)
        return False
    

# Takes a list of masked sentences and returns a dataframe containing the scores for all 
# masked token predictions
def compare_masks(masked_sentences, print_results=True, topk=20, model_names=['bert-base-uncased', 'roberta-base'], lemmatize=True):
    results = pd.DataFrame()
    for model_name in model_names:
        for sentence in masked_sentences:
            masks = get_mask(sentence, topk=topk, model_name=model_name, print_results=print_results, lemmatize=lemmatize)
            df = pd.DataFrame(masks, columns=["token", sentence])
            df = df.set_index("token")
            results = pd.concat((results, df), axis=1)

    return results


#Reads and parses dunlosky norms data and returns dictionary
def dunlosky_norms():
    raw = pd.read_csv("dunlosky_norms_unparsed.csv")
    parsed={}
    for n, line in raw.iterrows():
        if isinstance(line[0], str) and len(line[0].split((". "))) > 1:
            key = line[0].split((". "))[1]
            parsed[key] = {}
            print(n, key)
            k = 1
            while not (isinstance(raw.iloc[n+k, 0], str) and len(raw.iloc[n+k, 0].split((". "))) > 1) and n+k<len(raw)-1:
                if raw.iloc[n+k, 0] == 'Response' or isinstance(raw.iloc[n+k, 0], float):
                    print('null row')
                else:
                    if raw.iloc[n+k, 0][0] != '\xa0':
                        subkey = raw.iloc[n+k, 0]
                        parsed[key][subkey] = raw.iloc[n+k][1:].to_dict()
                        parsed[key][subkey]['variations'] = [raw.iloc[n+k, 0]]
                    else:
                        parsed[key][subkey]['variations'].append(raw.iloc[n+k, 0].replace('\xa0', ''))
                k+=1

    for cue in parsed:
        for response in parsed[cue]:
            variations = []
            print(parsed[cue][response]['variations'])
            for word in parsed[cue][response]['variations']:
                variations += [word.replace("(s)", ""), word.replace("(s)", "s"), wn.lemmatize(word)]
            parsed[cue][response]['variations'] += variations
            parsed[cue][response]['variations'] = list(set(parsed[cue][response]['variations']))
            print(parsed[cue][response]['variations'])

    return parsed

#Returns  associated masked phrases for dunlosky norms data
def dunlosky_masks():
    return ['a <MASK> is a precious stone',
    'a <MASK> is a unit of time',
    '<MASK> is a relative',
    'a <MASK> is a unit of distance',
    '<MASK> is a metal',
    'a <MASK> is a type of reading material',
    '<MASK> is a military title',
    'a <MASK> is a four-footed animal',
    'a <MASK> is a type of fabric',
    '<MASK> is a color',
    'a <MASK> is a kitchen utensil',
    'a <MASK> is a building for religious services',
    'a <MASK> is a part of speech',
    'a <MASK> is an article of furniture',
    'a <MASK> is a part of the human body',
    'a <MASK> is a fruit',
    'a <MASK> is a weapon',
    'a <MASK> is an elective office',
    'a <MASK> is a type of human dwelling',
    '<MASK> is an alcoholic beverage',
    '<MASK> is a country',
    '<MASK> is a crime',
    'a <MASK> is a carpenter tool',
    'a <MASK> is a member of the clergy',
    'a <MASK> is a substance for flavoring food',
    '<MASK> is a fuel',
    'a <MASK> is an occupation or profession',
    'a <MASK> is a natural earth formation',
    '<MASK> is a sport',
    'a <MASK> is a weather phenomenon',
    'a <MASK> is an article of clothing',
    'a <MASK> is a part of a building',
    '<MASK> is a chemical element',
    'a <MASK> is a musical instrument',
    'a <MASK> is a kind of money',
    '<MASK> is a type of music',
    'a <MASK> is a bird',
    'a <MASK> is a transportation vehicle',
    '<MASK> is a science',
    'a <MASK> is a toy',
    'the <MASK> is a type of dance',
    'a <MASK> is a vegetable',
    'a <MASK> is a type of footwear',
    'a <MASK> is an insect',
    'a <MASK> is a flower',
    '<MASK> is a disease',
    'a <MASK> is a tree',
    'a <MASK> is a type of ship or boat',
    'a <MASK> is a fish',
    'a <MASK> is a snake',
    '<MASK> is a city',
    '<MASK> is a state',
    '<MASK> is a drug',
    'a <MASK> is a type of car',
    '<MASK> is a liquid', 
    'a <MASK> is a thing women wear']


# Returns token scores for sentence with cases based on mask phrase formats
def get_token_scores(sentence, model='BERT', topk=100, lemmatize=True):
    if model=='BERT':
        if "a <MASK> is" in sentence:
            token_scores = {}
            phrases = ["a <MASK> is", "an <MASK> is", "<MASK> are"]
            k = int((topk/len(phrases)))
            while len(token_scores)<100:
                for phrase in phrases:
                    token_scores[phrase] = {token:score for (token, score) in get_mask(sentence.replace("a <MASK> is", phrase), show=False, topk=k, lemmatize=lemmatize) 
                                            if nlp(token)[0].pos_ in ['NOUN', 'VERB']}
                token_scores = dict_mean(token_scores[phrases[0]], token_scores[phrases[1]], token_scores[phrases[2]])
                k+=10
            token_scores = dict(sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:100])
            return token_scores
            
        else:
            token_scores = {token:score for (token, score) in get_mask(sentence, topk=topk, show=False, lemmatize=lemmatize) 
                            if nlp(token)[0].pos_ in ['NOUN', 'VERB']}
            return token_scores
    if model=='w2v':
        return w2v_getn(sentence)
    

# Compares list of token scores from model responses with norms data
def correct_responses(token_scores, data, category, limit = False):
    responses = data[category]
    correct = []
    missing = []
    if limit:
        responses = dict(sorted(responses.items(), key=lambda x: x[1]['Total'], reverse=True)[:limit])
    for response in responses:
        if [i for i in responses[response]['variations'] if i.lower() in list(token_scores.keys())]:
            correct.append(response)
        else:
            missing.append(response)
    return correct, missing, len(correct)/len(responses)

from itertools import chain, groupby
from operator import itemgetter
from collections import Counter


# Get max of each item from 2 dictionaries
def dict_max(data1, data2):
    get_key, get_val = itemgetter(0), itemgetter(1)
    merged_data = sorted(chain(data1.items(), data2.items()), key=get_key)
    merged = {k: max(map(get_val, g)) for k, g in groupby(merged_data, key=get_key)}
    norm = sum(merged.values())
    return {key:merged[key]/norm for key in merged.keys()}


# Get mean of items in 3 dictionaries, with weight=0 if item not in dict
def dict_mean(A,B,C={}):
    sums = dict(Counter(A) + Counter(B) + Counter(C))
#     print(sums)
    div=3
    if C=={}: # 2 dict case
        div = 2
    return {k:sums[k]/div for k in sums}

import re


# Preprocesses masked sentence into tokens list for w2v similarity analysis
# Helper function for w2v_getn
def w2v_pre(sentence):
    tokens = re.split('-| ', sentence.lower())
    tokens = [x.strip("():,.?").replace("\'", "") for x in tokens]
    for item in ['<mask>', 'youve', '', 'theyll']:
        if item in tokens:
            tokens.remove(item)
            
    tokens = [x for x in tokens if nlp(x)[0].pos_ in ['NOUN', 'VERB', 'ADJ']]
            
    return tokens


# Get topn similar tokens to sentence
def w2v_getn(sentence, topn=100, model=glove_vectors, lemmatize=True):
    tokens = w2v_pre(sentence)
    out = {}
    n = topn
    while len(out) < topn:
        if lemmatize:
            out = {wn.lemmatize(token):score for (token, score) in model.most_similar(tokens, topn=n) if nlp(token)[0].pos_ in ['NOUN', 'VERB']}
        else:
            out = {token:score for (token, score) in model.most_similar(tokens, topn=n) if nlp(token)[0].pos_ in ['NOUN', 'VERB']}
        n += 10
    
    if len(out) > 100:
        out = dict(sorted(out.items(), key=lambda x: x[1], reverse=True)[:topn])
        
    return out


def intersect(s1, s2):
    return [x for x in s1 if x in s2]
    