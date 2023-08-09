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

device = "cuda" if torch.cuda.is_available else cpu

from Glove import glove

glove_file = r"C:\Users\rahul\python\Machine learning\Data\glove.6B.50d.txt"
word_to_index, index_to_word, word_to_vec_map = glove(glove_file)

from Important_to_convert import sentence_to_index,label_to_emoji,pretrained_embedded_layer

from Data_set import Custom_dataset

train_sample_dir = r"C:\Users\rahul\python\Machine learning\Data\train_emoji.csv"
test_sample_dir = r"C:\Users\rahul\python\Machine learning\Data\test_emoji.csv" 

train_dataset = Custom_dataset(train_sample_dir)
train_dataLoader = DataLoader(dataset=train_dataset,batch_size=16,shuffle=False)

test_dataset = Custom_dataset(test_sample_dir)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=False)

from model import Emojify__Rnn

model = Emojify__Rnn(128,2,False,word_to_vec_map,word_to_index,pretrained_embedded_layer).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.01)

from train_and_test import train,test

def model_train():
    Epoch = 50
    train_loss_list = []
    train_acc_list = []

    test_loss_list = []
    test_acc_list = []
    for epoch in tqdm(range(Epoch)):
        train_loss,train_acc = train(model,train_dataLoader,loss_fn,optimizer,device)
        train_loss_list.append(train_loss.item())
        train_acc_list.append(train_acc)
        test_loss,test_acc = test(model,test_dataloader,loss_fn,device)
        test_loss_list.append(test_loss.item())
        test_acc_list.append(test_acc)
        if epoch % 10 ==0:
            print(f"Epoch__ {epoch}")
            print(f"Train Loss: {train_loss:.2f} | Train Accuracy : {train_acc:.2f}")
            print(f"Validation Loss: {test_loss:.2f} | Validation Accuracy : {test_acc:.2f}")
            print() 
            
    plt.subplot(2,1,1)
    plt.title("Loss")
    plt.plot(range(Epoch),train_loss_list,label = "Train loss")
    plt.plot(range(Epoch),test_loss_list, label = "Validation loss")


    plt.subplot(2,1,2)
    plt.title("Accuracy")
    plt.plot(range(Epoch),train_acc_list,label = "Train Accuracy")
    plt.plot(range(Epoch),test_acc_list, label = "Validation Accuracy") 

    plt.tight_layout()