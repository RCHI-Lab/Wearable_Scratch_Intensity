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
from bisect import bisect
import random
import datetime 
import gc 

sys.path.append("/home/akhil/Documents/research/final_code/")
import processing_functions as pf

fh_output = '/home/akhil/Documents/research/final_code/results/intensity/'
def write(output):
    file_object = open(fh_output + 'intensity_acc.txt', 'a')
    file_object.write(output)
    file_object.write("\n")
    file_object.close()
def write_newline():
    file_object = open(fh_output + 'intensity_acc.txt', 'a')
    file_object.write("\n")
    file_object.close()





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


def my_loss(y, preds, weights):
    loss = torch.abs(y - preds)*torch.from_numpy(np.array(weights))
    loss = torch.mean(loss)
    return loss

def percent_loss(y, preds):
    loss = torch.abs(y - preds)
    loss = torch.div(loss, y)*100
    loss = torch.mean(loss)
    return loss
def percent_loss600(y, preds):
    loss = torch.abs(y - preds)
    loss = torch.div(loss, 600)*100
    loss = torch.mean(loss)
    return loss



#import data 
filename = '/home/akhil/Documents/research/final_code/data/cleaned_intensity_data_25.0_1.pickle'
d = pd.read_pickle(filename)
wearable_data = np.array(d['wearable_data'])
labels = np.array(d['power_labels'])
loso_labels = d['loso_labels']
X = wearable_data
y = labels



hidden_layer_dim = [1000]
num_hidden_layers = [2]
dropout = [0.1]
con_crop = [400]
acc_crop =[175]


logo = LeaveOneGroupOut()
k = logo.get_n_splits(groups = loso_labels)
samp_len_secs = 1
samp_freq1 = 8000
samp_freq2 = 400
sample_len1 = int(samp_freq1*samp_len_secs)
sample_len2 = int(samp_freq2*samp_len_secs)
preprocessing = ['n']

grid = {'hidden_layer_dim': hidden_layer_dim,
        'num_hidden_layers': num_hidden_layers,
        'dropout': dropout,
        'con_crop' : con_crop,
        'acc_crop' : acc_crop,}

start_time = time.time()
results = []
epochs_min_args = []
all_params = []
combs = list(itertools.product(*grid.values()))
random.shuffle(combs)
n = 1
for params in combs:
    write_newline()
    output = str(n) + " " + str(int(time.time()-start_time)) + " " + str(params)
    print(output)
    write(output)

    hidden_layer_dim, num_hidden_layers, dropout, con_crop, acc_crop = params 

    all_train_loss = []
    all_test_loss = []
    all_test_lossmae = []
    p_counter = 1
    all_train_losses = []
    all_test_losses = []
    all_labels = []
    all_predictions = []
    naive_predictor_mean = []
    for train_index, test_index in logo.split(X, y, loso_labels):
        print("Participant", str(p_counter))
        part_train_losses = []
        part_test_losses = []
        part_test_losses2 = []

        #acc or con only 
        start_dim = acc_crop
        #start_dim = con_crop

        #start_dim = acc_crop + con_crop
        mlp = MLP_grid(start_dim, hidden_layer_dim, num_hidden_layers, dropout)
        loss_function = nn.L1Loss()

        lr = 5e-6
        optimizer = torch.optim.Adam(mlp.parameters(), lr= lr)#1e-1)
        batch_size = 64 
            
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        X_train, X_test = pf.apply_fft(X_train, X_test, con_crop, acc_crop, sample_len1, sample_len2)

        for p in preprocessing:
            if p =='s':
                X_train, X_test = pf.standardize(X_train, X_test)
            elif p == 'n':
                X_train, X_test, normalizer = pf.normalize(X_train, X_test)

        #take only accelerometer values 
        X_train = X_train[:,con_crop:]
        X_test = X_test[:,con_crop:]

        #take only contact microphone values 
        #X_train = X_train[:,0:con_crop]
        #X_test = X_test[:,0:con_crop]

        
        rand_inds = np.random.permutation(np.shape(X_train)[0])
        X_train = X_train[rand_inds]
        y_train = y_train[rand_inds]

        naive_predictor_mean.append((np.mean(y_train)))

        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()
        epochs = 171
        for epoch in range(1,epochs+1):
            #print(f'Starting epoch {epoch+1}')
            current_loss = 0.0
            mlp.train()
            for (batchX, batchy) in next_batch(X_train, y_train, batch_size):
                optimizer.zero_grad()
                outputs = mlp(batchX)
                outputs = torch.squeeze(outputs)
                loss = loss_function(batchy, outputs)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()*batchy.size(0)

            with torch.no_grad():
                mlp.eval()
                predictions = mlp(X_test)
                predictions = torch.squeeze(predictions)
                loss = loss_function(y_test, predictions)
                testLoss = loss.item()
            avg_loss = current_loss / np.shape(X_train)[0]
            part_train_losses.append(avg_loss)
            part_test_losses.append(testLoss)
            print("Epoch Number:", epoch, "Train Acc:", avg_loss, "Test Acc:", testLoss)
        output = "Participant " + str(p_counter) + "  " + str(avg_loss) + ' '+ str(testLoss)
        write(output)
        p_counter+=1
        all_train_loss.append(avg_loss)
        all_test_loss.append(testLoss)
        all_train_losses.append(part_train_losses)
        all_test_losses.append(part_test_losses)
        all_labels.append(y_test.numpy().tolist())
        all_predictions.append(predictions.numpy().tolist()) 
        
       
    output = "Avg Train Loss: " + str(np.mean(all_train_loss)) + "   Avg Test Loss: " + str(np.mean(all_test_loss))
    write(output)
    write_newline()
    print(output)

    #naive predictor errors 
    naive_mean_errors = []
    naive_median_errors = []
    naive_mape_errors = []
    naive_mdape_errors = []
    for p in np.arange(0, 20):
        pred = naive_predictor_mean[p]
        l = all_labels[p]
        e = np.mean(np.abs(np.array(l)-pred))
        naive_mean_errors.append(e)
        e = np.median(np.abs(np.array(l)-pred))
        naive_median_errors.append(e)
        e = 100*np.mean(np.abs(np.array(l)-pred)/np.array(l))
        naive_mape_errors.append(e)
        e = 100*np.median(np.abs(np.array(l)-pred)/np.array(l))
        naive_mdape_errors.append(e)

    print("Naive Prediction Mean:", np.mean(naive_mean_errors), np.std(naive_mean_errors))
    print("Naive Prediction Median:", np.mean(naive_median_errors), np.std(naive_median_errors))
    print("Naive Prediction MAPE:", np.mean(naive_mape_errors), np.std(naive_mape_errors))
    print("Naive Prediction MDAPE:", np.mean(naive_mdape_errors), np.std(naive_mdape_errors))

    mean_errors = []
    median_errors = []
    mape_errors = []
    mdape_errors = []
    for p in np.arange(0,20):
        l = all_labels[p]
        p = all_predictions[p]
        e = np.mean(np.abs(np.array(l)-np.array(p)))
        mean_errors.append(e)
        e = np.median(np.abs(np.array(l)-np.array(p)))
        median_errors.append(e)
        e = 100*np.mean(np.abs(np.array(l)-np.array(p))/np.array(l))
        mape_errors.append(e)
        e = 100*np.median(np.abs(np.array(l)-np.array(p))/np.array(l))
        mdape_errors.append(e)
    #print("All Errors:", errors)
    mae = np.mean(mean_errors)
    std = np.std(mean_errors)
    print("Mean Mean Absolute Error:", np.round(mae, 4), "STD:", np.round(std,4))
    mae = np.mean(median_errors)
    std = np.std(median_errors)
    print("Mean Median Absolute Error:", np.round(mae, 4), "STD:", np.round(std,4))
    mape = np.mean(mape_errors)
    std = np.std(mape_errors)
    print("Mean Mean Absolute Percentage Error:", np.round(mape, 4), "STD:", np.round(std,4))
    mdape = np.mean(mdape_errors)
    std = np.std(mdape_errors)
    print("Mean Median Absolute Percentage Error:", np.round(mdape, 4), "STD:", np.round(std,4))




    epoch_array = np.arange(1,epochs+1)
    train_losses_avg = np.mean(all_train_losses, axis = 0)
    test_losses_avg = np.mean(all_test_losses, axis = 0)

    min_test_loss =  np.min(test_losses_avg)
    output = "Minimum Test Loss: " + str(min_test_loss)
    write(output)
    min_epoch = np.argmin(test_losses_avg)+1
    output = "Epoch Number: " + str(min_epoch)
    write(output)

    #output labels/predictions
    filename = fh_output +"intensity_acc_labelpreds.pickle"
    d = {'labels' : all_labels, 'predictions': all_predictions}

    with open(filename, 'wb') as handle:
        pkl.dump(d, handle)


    #print(np.shape(epoch_array))

    plt.plot(epoch_array, train_losses_avg)
    plt.plot(epoch_array, test_losses_avg)
    plt.legend(['train loss', 'test loss'])
    plt.xlabel('Epochs')
    plt.ylabel('MAE (mW)')
    #plt.show()
    plot_title = "Intensity Regression Accelerometer Only" 
    plt.title(plot_title)

    plt.savefig(fh_output + "intensity_acc" + ".png")
    plt.clf()
    write(output)
    n+=1
