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

def glove(glove_file):
    with open(glove_file, 'rb') as f:
        words = []
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            byte_string = curr_word
            curr_word = byte_string.decode('utf-8')
            words.append(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map