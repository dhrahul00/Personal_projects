import torch
from torch import nn
import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#%matplotlib inline
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available else cpu

from Glove import glove

glove_file = r"C:\Users\rahul\python\Machine learning\Data\glove.6B.50d.txt"
word_to_index, index_to_word, word_to_vec_map = glove(glove_file)

emoji_dictionary = {
    0:'❤️',
    1:'⚾',
    2:'😄',
    3:'😔',
    4:'🍴',
}

def label_to_emoji(label):
    return emoji_dictionary[label]


def sentence_to_index(array_word,word_to_index,max_len):
    indices = torch.zeros(max_len)
    for idx,word in enumerate(array_word.lower().split()):
        if word in word_to_index and idx < max_len:
            indices[idx] = word_to_index[word]
    return indices.long()

def pretrained_embedded_layer(word_to_vec_map,word_to_index):
    
    num_embeddings = len(word_to_vec_map) +1
    embedding_dim = len(word_to_vec_map[list(word_to_vec_map.keys())[0]])
    
    embeddings = torch.zeros(num_embeddings,embedding_dim)
    
    for word,idx in word_to_index.items():
        embeddings[idx,:] = torch.from_numpy(word_to_vec_map[word])
        
    embedding_layer = nn.Embedding.from_pretrained(embeddings.to(device),freeze=True)
    
    return embedding_layer