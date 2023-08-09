import torch
from torch import nn
import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

class Emojify__Rnn(nn.Module):
    def __init__(self,hidden_size,num_layers,bidirectional,word_to_vec_map,word_to_index,pretrained_embedded_layer):
        super().__init__()
        if bidirectional:
            bi_value = 2
        else:
            bi_value = 1
        embedding_layer = pretrained_embedded_layer(word_to_vec_map,word_to_index)
        self.embedding_layer = embedding_layer
        self.packed_padded_seq = nn.utils.rnn.pack_padded_sequence
        self.pad_packed = nn.utils.rnn.pad_packed_sequence
        self.lstm1 = nn.LSTM(input_size=50,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_layers,
                            bidirectional = bidirectional,
                            dropout=0.5)
        self.linear1 = nn.Linear(hidden_size,64)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64,5)
        self.apply(self._init_weights)
        
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        if isinstance(layer, nn.LSTM):
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0.01)
        
    
    def forward(self,X):
        lengths = torch.count_nonzero(X,dim=1).cpu()
        total_length = X.size(-1)
        X = self.embedding_layer(X)
        X = self.packed_padded_seq(input=X,lengths=lengths,batch_first=True,enforce_sorted=False)
        X,_ = self.lstm1(X)
        X,_ = self.pad_packed(X,batch_first=True,padding_value = 0,total_length = total_length )
        X = X[torch.arange(X.size(0)), lengths - 1, :]
        X = self.linear1(X)
        X = self.dropout(X)
        X = self.linear2(X)
        
        return X