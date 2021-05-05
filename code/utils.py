import pickle
import tqdm
import torch
import gensim
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from torchcrf import CRF
from config import *

tag2idx = {'S': 0, 'B': 1, 'M': 2, 'E': 3}
idx2tag = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}
PAD_IDX = 0
UNK_IDX = 1
pad_vec = np.random.normal(size = 300).reshape(1, -1)
unk_vec = np.random.normal(size = 300).reshape(1, -1)

words2idx = {}
idx2words = {}
words_list = ['<pad>', '<unk>']
word_vectors = []

def write_result(epoch, result, file_name):
    f = open(file_name, 'a')
    f.write(str(epoch))
    f.write('\n')
    f.write("-loss: %.4f  -F1: %.4f  -precision: %.4f  -recall: %.4f" %result[-1])
    f.write('\n')
    f.close()

def data_cleaning(df, is_test = False):
    df['sentence'] = df['original'].apply(lambda x: x.replace(' ', ''))
    punctuations = []
    for item in punctuations:
        df['original'] = df['original'].apply(lambda x: x.replace(item, ''))
        df['sentence'] = df['sentence'].apply(lambda x: x.replace(item, ''))
    
    df['words'] = df['original'].apply(lambda x: [a for a in x.split(' ') if a != '' ])
    if is_test == False:
        df['length'] = df['sentence'].apply(lambda x: len(x))
        df = df.drop(df[df['length'] > pad_size].index)
        df = df.drop(df[df['length'] <= 5].index) # there are empty sentences in the dataset!
        df = df.sort_values(by = 'length')
        df = df.drop('length', axis = 1)
        df = df.reset_index(drop = True)
    return df

def tag(words):
    tag_list = []
    for word in words:
        if len(word) == 1:
            tag_list.append(tag2idx['S'])
        else:
            tag_list.append(tag2idx['B'])
            tag_list += [tag2idx['M'] for k in range(len(word) - 2)]
            tag_list.append(tag2idx['E'])
    return tag_list
	
def tag_to_words(sentence, test_predict_gold):
    segmentation_list = []
    for i, item in enumerate(test_predict_gold):
        segmentation_list.append(sentence[item[0]:item[1]+1])
    segmentation = ' '.join(segmentation_list)
    return segmentation

def id_and_pad_sentence(sentence): # and pad
    idx_list = []
    for char in sentence:
        if char in words_list:
            idx_list.append(words2idx[char])
        else:
            idx_list.append(UNK_IDX)
    return idx_list
	
def collate_fn(batch):
    batch.sort(key = lambda x: len(x[0]), reverse = True)
    sentence = [item[0] for item in batch]
    tags = [item[1] for item in batch]
    length_list = [len(x[0]) for x in batch]
    padded_sentence = rnn_utils.pad_sequence(sentence, batch_first = True, padding_value = PAD_IDX)
    padded_tags = rnn_utils.pad_sequence(tags, batch_first = True, padding_value = tag2idx['S'])
    return padded_sentence, padded_tags, length_list
	
class MSRSEG(Dataset):
    def __init__(self, df):
        self.df = df
        self.df['tags'] = self.df['words'].apply(tag)
        self.df['sentence_idx'] = self.df['sentence'].apply(id_and_pad_sentence)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sentence_idx = self.df['sentence_idx'].iloc[idx]
        tags = self.df['tags'].iloc[idx]
        return torch.LongTensor(np.array(sentence_idx)), torch.LongTensor(np.array(tags))

def to_words_list(L):
    words_list = []
    start = 0
    while start < len(L):
        end = start + 1
        if idx2tag[L[start]] == 'S':
            words_list.append((start, start))
            start += 1
        elif idx2tag[L[start]] == 'B':
            while end < len(L) and idx2tag[L[end]] == 'M':
                end += 1
            words_list.append((start, end))
            start = end + 1
        else:
            start += 1
    return words_list

def evaluate(output, tags, length_list):
    tags = tags.cpu().numpy().tolist()
    num_same_words = 0
    num_output_words = 0
    num_tags_words = 0
    predict_words = []
    for i, item in enumerate(length_list):
        output[i] = output[i][:item]
        tags[i] = tags[i][:item]
        
        predict_words.append(to_words_list(output[i]))
        output_words_list = set(predict_words[-1])
        tags_words_list = set(to_words_list(tags[i]))
        
        num_same_words += len(output_words_list & tags_words_list)
        num_output_words += len(output_words_list)
        num_tags_words += len(tags_words_list)

    precision = 0 if num_output_words == 0 else num_same_words / num_output_words
    recall = 0 if num_tags_words == 0 else num_same_words / num_tags_words
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    return F1, precision, recall, predict_words
