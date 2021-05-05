import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from torchcrf import CRF
from utils import *
from config import *

class biLSTM_CNN_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, tag_to_idx, dropout_rate, word_embeddings):
        super(biLSTM_CNN_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.tag_to_idx = tag_to_idx
        self.tagset_size= len(tag_to_idx)
        
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=PAD_IDX)
        self.embedding.from_pretrained(torch.from_numpy(word_embeddings), freeze=True)
        
        self.biLSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True,bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first = True)
        
        self.init_linears()
        self.dropout = nn.Dropout(dropout_rate)
    
    def init_linears(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, sentence, tags, length_list):
        embeds = self.embedding(sentence)
        packed_embeds = rnn_utils.pack_padded_sequence(input=embeds,
                                                       lengths=length_list,
                                                       batch_first=True,
                                                       enforce_sorted=True)
        lstm_output, _ = self.biLSTM(packed_embeds)
        lstm_output, _ = rnn_utils.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_output = self.dropout(lstm_output)
        lstm_output = self.linear(lstm_output)
        
        emissions = lstm_output
        loss = -self.crf(emissions, tags)
        tag_seq_list = self.crf.decode(emissions)
        
        return tag_seq_list, loss