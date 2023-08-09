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



def train(model,dataloader,loss_fn,optimizer,device):
    model.train()
    
    total_loss = 0
    total_acc = 0
    
    for sentence,label in (dataloader):
        sentence = sentence.to(device)
        label = label.to(device)
        
        # forward pass :
        logits = model(sentence)
        label_pred = nn.Softmax(dim=1)(logits)
        
        # loss & Accuracy :
        loss = loss_fn(logits,label)
        acc = (label_pred.argmax(dim=1) == label).sum().item()/(label.size(0))
        
        total_loss += loss
        total_acc += acc
        
        # optimizer zero grad:
        optimizer.zero_grad()
        
        # backward pass:
        loss.backward()
        
        # optimizer step:
        optimizer.step()
        
    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    
    return total_loss,total_acc

def test(model,dataloader,loss_fn,device):
    model.eval()
    
    total_loss = 0
    total_acc = 0
    
    for sentence,label in dataloader:
        sentence,label = sentence.to(device),label.to(device)
        
        # forward pass:
        logits = model(sentence)
        label_pred = nn.Softmax(dim=1)(logits)
        
        # loss & accuracy:
        loss = loss_fn(logits,label)
        acc = (label_pred.argmax(dim=1) == label).sum().item()/(label.size(0))
        
        total_loss += loss
        total_acc += acc
    
    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    
    return total_loss,total_acc