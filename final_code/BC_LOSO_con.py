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

fh_output = '/home/akhil/Documents/research/final_code/results/bc/'
def write(output):
    file_object = open(fh_output + 'bc_con.txt', 'a')
    file_object.write(output)
    file_object.write("\n")
    file_object.close()
def write_newline():
    file_object = open(fh_output + 'bc_con.txt', 'a')
    file_object.write("\n")
    file_object.close()



def next_batch(inputs, targets, batchSize):
    # loop over the dataset
    for i in range(0, inputs.shape[0], batchSize):
        # yield a tuple of the current batched data and labels
        yield (inputs[i:i + batchSize], targets[i:i + batchSize])        


class Net(nn.Module):
    def __init__(self, start_dim, hidden_dim, out_dim, num_hidden_layers, dropout, actfunc):
        super(Net,self).__init__()
        self.start_dim = start_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim 
        self.fc1 = nn.Linear(self.start_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.out_dim)

        self.layers = nn.ModuleList()
        self.layers.append(self.fc1)
        for n in np.arange(0, num_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(self.out)

        self.drop = nn.Dropout(p=dropout)
        self.actfunc = nn.ReLU() 
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.drop(self.actfunc(layer(x)))
        x = self.layers[-1](x)
        x = self.sig(x)
        return x 





#import data 
fileheader = '/home/akhil/Documents/research/final_code/data/'
name = fileheader + "bcdata_25.0_" + str(1) + ".pickle"
d = pd.read_pickle(name)
wearable_data = d['X']
labels = d['y']
loso_labels = d['loso_labels']
cat_labels = d['cat_labels']
X = wearable_data
y = labels


num_pca_components = [0]
hidden_layer_dim = [1200]
num_hidden_layers = [3]
dropout = [0.2]
epochs = [200]
activation = ['relu']
batch_size = [64]
preprocessing = [['n']]
con_crop = [275]
acc_crop = [200]
#samp_lens = [1.25, 1.5, 1.75, 2, 1]
samp_lens = [1.0]

grid = {'num_pca_components': num_pca_components,
                'hidden_layer_dim': hidden_layer_dim,
                'num_hidden_layers': num_hidden_layers,
                'dropout': dropout,
                'epochs': epochs,
                'activation': activation,
                'batch_size' : batch_size,
                'preprocessing' : preprocessing,
                'con_crop' : con_crop,
                'acc_crop' : acc_crop,
                'samp_len' : samp_lens}




start_time = time.time()
results = []
epochs_min_args = []
all_params = []
combs = list(itertools.product(*grid.values()))
random.shuffle(combs)
n = 2
for params in combs:
    #print(" ")
    write_newline()
    output = str(n) + " " + str(int(time.time()-start_time)) + " " + str(params)
    print(output)
    write(output)

    _, hidden_layer_dim, num_hidden_layers, dropout, _,actfunc, batch_size, preprocessing, con_crop, acc_crop, samp_len = params 

    logo = LeaveOneGroupOut()
    k = logo.get_n_splits(groups = loso_labels)
    samp_freq1 = 8000
    samp_freq2 = 400
    sample_len1 = int(samp_freq1*samp_len)
    sample_len2 = int(samp_freq2*samp_len)
    all_train_loss = []
    all_test_loss = []
    all_test_lossmae = []
    p_counter = 1
    all_train_acc = []
    all_test_acc = []
    all_labels = []
    all_predictions = []
    all_cat_labels = []
    for train_index, test_index in logo.split(X, y, loso_labels):
        print("Participant", str(p_counter))
        part_train_losses = []
        part_test_losses = []
        #start_dim = acc_crop + con_crop
        #start_dim = acc_crop
        start_dim = con_crop


        actfunc = torch.relu
        mlp = Net(start_dim, hidden_layer_dim, 1, num_hidden_layers, dropout, actfunc)
        #mlp = Net(start_dim, 200, 1, 1, 0, '')
        #loss_function = nn.L1Loss()
        loss_function = nn.BCELoss()
        #loss_function = percent_loss

        lr =1e-5
        optimizer = torch.optim.Adam(mlp.parameters(), lr= lr)#1e-1)
        batch_size = 64
            
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        c_labels_train, c_labels_test = cat_labels[train_index], cat_labels[test_index]
        X_train, X_test = pf.apply_fft(X_train, X_test, con_crop, acc_crop, sample_len1, sample_len2)


        for p in preprocessing:
            if p =='s':
                X_train, X_test = pf.standardize(X_train, X_test)
            elif p == 'n':
                X_train, X_test, normalizer = pf.normalize(X_train, X_test)

        #take only accelerometer values 
        #X_train = X_train[:,con_crop:]
        #X_test = X_test[:,con_crop:]

        #take only contact microphone values 
        X_train = X_train[:,0:con_crop]
        X_test = X_test[:,0:con_crop]


        
        rand_inds = np.random.permutation(np.shape(X_train)[0])
        X_train = X_train[rand_inds]
        y_train = y_train[rand_inds]

        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()
        epochs = 144
        train_acc_epochs = []
        test_acc_epochs = []
        for epoch in np.arange(1,epochs+1):
            #print(f'Starting epoch {epoch+1}')
            current_loss = 0.0
            samples = 0
            trainAcc = 0
            mlp.train()

            for (batchX, batchy) in next_batch(X_train, y_train, batch_size):
                optimizer.zero_grad()
                outputs = mlp(batchX)
                outputs = torch.squeeze(outputs)
                loss = loss_function(outputs, batchy)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()*batchy.size(0)
                curr_batch_size = batchy.size(0)
                trainAcc += torch.sum(torch.round(outputs) == batchy)
                samples += curr_batch_size
            trainLossEpoch = current_loss/samples
            trainAccEpoch= trainAcc/samples


            with torch.no_grad():
                mlp.eval()
                predictions = mlp(X_test)
                predictions = torch.squeeze(predictions)
                loss = loss_function(predictions, y_test)
                testLoss = loss.item() 
                test_size = y_test.size(0)
                testAccEpoch = torch.sum(torch.round(predictions) == y_test)/test_size
            trainAccEpoch = trainAccEpoch.item()
            testAccEpoch = testAccEpoch.item()
            train_acc_epochs.append(trainAccEpoch)
            test_acc_epochs.append(testAccEpoch)
            print("Epoch Number:", epoch, "Train Acc:", trainAccEpoch, "Test Acc:", testAccEpoch)
            
        output = "Participant " + str(p_counter) + "  " + str(trainAccEpoch) + ' '+ str(testAccEpoch)
        write(output)
        p_counter+=1
        all_train_acc.append(train_acc_epochs)
        all_test_acc.append(test_acc_epochs)
        all_labels.append(y_test.numpy().tolist())
        all_predictions.append(predictions.numpy().tolist())  
        all_cat_labels.append(c_labels_test)

 
    output = "Avg Train Loss: " + str(np.mean(np.array(all_train_acc)[:,-1])) + "   Avg Test Loss: " + str(np.mean(np.array(all_test_acc)[:,-1]))
    write(output)
    write_newline()

    print("Mean Test Loss and STD:", np.mean(np.array(all_test_acc)[:,-1]), np.std(np.array(all_test_acc)[:,-1]))


    #output labels/predictions
    filename = fh_output +"bc_labelpreds_con.pickle"
    d = {'labels' : all_labels, 'predictions': all_predictions, 'cat_labels' : all_cat_labels}

    with open(filename, 'wb') as handle:
        pkl.dump(d, handle)

    epoch_array = np.arange(1,epochs+1)
    train_losses_avg = np.mean(all_train_acc, axis = 0)
    test_losses_avg = np.mean(all_test_acc, axis = 0)
    test_losses_std = np.std(all_test_acc, axis = 0)

    max_acc =  np.max(test_losses_avg)
    output = "Max Test Accuracy: " + str(max_acc)
    write(output)
    min_epoch = np.argmax(test_losses_avg)
    output = "Epoch Number: " + str(min_epoch+1)
    write(output)
    max_std = test_losses_std[min_epoch]  
    output = "Max Accuracy: " +  str(max_acc) + "  Std: " + str(max_std) + "  Epoch:" + str(min_epoch+1)
    write(output)
    print("Max Accuracy:", str(max_acc), "Std:", str(max_std), "Epoch:", str(min_epoch+1))





    #print(np.shape(epoch_array))
    plt.plot(epoch_array[0:min_epoch+1], train_losses_avg[0:min_epoch+1])
    plt.plot(epoch_array[0:min_epoch+1], test_losses_avg[0:min_epoch+1])
    plt.legend(['train acc', 'test acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.show()

    plot_title = "Binary Classification Con. Mic. Only" 
    plt.title(plot_title)

    plt.savefig(fh_output + "bc_con" + ".png")
    plt.clf()
    write(output)
    n+=1

