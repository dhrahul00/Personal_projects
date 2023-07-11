# In predict.py
import torch
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from helper import ResNet_18

model = ResNet_18(4, 1)

model.load_state_dict(torch.load(r"/content/ResNet_18.pth"))



def check():
    print(0)


def decaptcha(filepaths):
    # Function implementation here
    X = np.zeros((len(filepaths),100,500,4))
    for idx,image in enumerate(filepaths):
        i_image = Image.open(image)
        i_image = np.array(i_image)
        X[idx] = i_image


    tensor = torch.from_numpy(X).type(torch.float)
    tensor = torch.transpose(tensor,3,2)
    tensor = torch.transpose(tensor,2,1)
    tensor /= 255.0
    
    model.eval() 
    with torch.inference_mode():
        y_pred = model(tensor)
        y_pred = torch.round(torch.nn.Sigmoid()(y_pred))
    
    y_pred = y_pred.numpy()
    result = np.where(y_pred == 1 ,"ODD","EVEN")
    
    return result.squeeze() 