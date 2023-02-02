import numpy as np 
import pickle as pkl
import pandas as pd 
import matplotlib.pyplot as plt

#processing imports 
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq

#ML imports 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import Normalizer

import sys

#multiprocessing
from joblib import Parallel, delayed 
import time  
import torch
from torch import nn
from torch.nn import functional as F
import itertools
import joblib

sys.path.append("/home/akhil/Documents/research/old/study/")
import processing_functions as pf

#import data 
filename = '/home/akhil/Documents/research/final_code/data/cleaned_intensity_data_25.0_1.pickle'
d = pd.read_pickle(filename)
wearable_data = np.array(d['wearable_data'])
labels = np.array(d['power_labels'])
loso_labels = d['loso_labels']
X = wearable_data
y = labels





def next_batch(inputs, targets, batchSize):
    # loop over the dataset
    for i in range(0, inputs.shape[0], batchSize):
        # yield a tuple of the current batched data and labels
        yield (inputs[i:i + batchSize], targets[i:i + batchSize])        

class MLP_grid(nn.Module):
    def __init__(self, start_dim, hidden_dim, num_hidden_layers, dropout):
        super(MLP_grid,self).__init__()
        self.start_dim = start_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.start_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()

        self.layers = nn.ModuleList()
        self.layers.append(self.fc1)
        for n in np.arange(0, num_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(self.out)

        self.drop = nn.Dropout(p=dropout)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.drop(x)
        x = self.layers[-1](x)
        return x 


logo = LeaveOneGroupOut()
k = logo.get_n_splits(groups = loso_labels)
samp_len_secs = 1
samp_freq1 = 8000
samp_freq2 = 400
sample_len1 = int(samp_freq1*samp_len_secs)
sample_len2 = int(samp_freq2*samp_len_secs)
preprocessing = ['n']

all_train_loss = []
all_test_loss = []
p_counter = 1


hidden_layer_dim = 1000
num_hidden_layers = 2
dropout = 0.1
con_crop = 400
acc_crop = 175

#start_dim = con_crop + acc_crop
start_dim = acc_crop
mlp = MLP_grid(start_dim, hidden_layer_dim, num_hidden_layers, dropout)
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr= 5e-6)#1e-1)
batch_size = 64 
    
X_test = X[0:2, :]
X_train, X_test = pf.apply_fft(X, X_test, con_crop, acc_crop, sample_len1, sample_len2)

#take only accelerometer values 
X_train = X_train[:,con_crop:]
X_test = X_test[:,con_crop:]


for p in preprocessing:
    if p =='s':
        X, X_test = pf.standardize(X_train, X_test)
    elif p == 'n':
        X, X_test, normalizer = pf.normalize(X_train, X_test)

joblib.dump(normalizer, '/home/akhil/Documents/research/final_code/intensity_normalizer_acc.gz')

rand_inds = np.random.permutation(np.shape(X_train)[0])
X = X[rand_inds]
y = y[rand_inds]
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
epochs = 171
for epoch in range(1,epochs+1):
    current_loss = 0.0
    mlp.train()
    for (batchX, batchy) in next_batch(X, y, batch_size):
        optimizer.zero_grad()
        outputs = mlp(batchX)
        outputs = torch.squeeze(outputs)

        loss = loss_function(outputs, batchy)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()*batchy.size(0)


    avg_loss = current_loss / np.shape(X_train)[0]
    print("Epoch:", epoch, "   Train Loss: ", avg_loss)

torch.save(mlp, '/home/akhil/Documents/research/final_code/intensity_model_acc')