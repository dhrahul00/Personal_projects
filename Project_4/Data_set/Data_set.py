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
from torch.utils.data import DataLoader,Dataset

from Important_to_convert import sentence_to_index,label_to_emoji,pretrained_embedded_layer
from Glove import glove

glove_file = r"C:\Users\rahul\python\Machine learning\Data\glove.6B.50d.txt"
word_to_index, index_to_word, word_to_vec_map = glove(glove_file)


class Custom_dataset(Dataset):
    def __init__(self,sample_dir):
        self.sample_dir = sample_dir
        self.df = pd.read_csv(sample_dir,names=["Sentence","label"],skiprows=1)
        self.df["len"] = self.df.Sentence.apply(self.len_check)
        self.max_len = max(self.df.len)
        
    def __len__(self):
        #print(self.df)
        return len(self.df)
    
    def len_check(self,x):
        return len(x.split())
    
    def __getitem__(self,index):
        sample = self.df.Sentence.iloc[index]
        label = self.df.label.iloc[index]
        sample = sentence_to_index(sample,word_to_index,self.max_len)
        return sample,label