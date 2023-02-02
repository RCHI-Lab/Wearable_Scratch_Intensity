import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random
import pickle 


#processing imports 
from scipy.interpolate import interp1d 
from scipy.fft import fft, fftfreq

import sys 
sys.path.append("/home/akhil/Documents/research/final_code/")
import processing_functions as pf


def bc_save_data(shift, samp_len_secs):
    num_secs = 30.0
    samp_freq1 = 8000
    samp_freq2 = 400
    wearable_data, labels, loso_labels, cat_labels = pf.get_bc_data(num_secs, shift, samp_freq1,samp_freq2, samp_len_secs)

    d = {'X' : wearable_data, 'y': labels, 'loso_labels' : loso_labels, 'cat_labels' : cat_labels}

    fh = '/home/akhil/Documents/research/final_code/data/'
    name = fh + "bcdata_" + str(shift*100) + "_" + str(samp_len_secs) + '.pickle'

    with open(name, 'wb') as handle:
        pickle.dump(d, handle)

def intensity_save_data_clean(shift, samp_len_secs): 
    num_secs = 10.0
    samp_freq1 = 8000
    samp_freq2 = 400
    wearable_data, force_labels, velocity_labels, power_labels, loso_labels = pf.get_intensity_data_clean(num_secs, shift, samp_freq1, samp_freq2, samp_len_secs)
    
    
    d = {'wearable_data': wearable_data, 'force_labels':force_labels, 'loso_labels':loso_labels, 
        'velocity_labels': velocity_labels, 'power_labels': power_labels}

    fh = '/home/akhil/Documents/research/final_code/data/'
    name = fh + "cleaned_intensity_data_" + str(shift*100) + "_" + str(samp_len_secs) + '.pickle'

    with open(name, 'wb') as handle:
        pickle.dump(d, handle)

def intensity_save_data_unclean(shift, samp_len_secs): 
    num_secs = 10.0
    samp_freq1 = 8000
    samp_freq2 = 400
    wearable_data, force_labels, velocity_labels, power_labels, loso_labels = pf.get_intensity_data_unclean(num_secs, shift, samp_freq1, samp_freq2, samp_len_secs)
    
    
    d = {'wearable_data': wearable_data, 'force_labels':force_labels, 'loso_labels':loso_labels, 
        'velocity_labels': velocity_labels, 'power_labels': power_labels}

    fh = '/home/akhil/Documents/research/final_code/data/'
    name = fh + "uncleaned_intensity_data_" + str(shift*100) + "_" + str(samp_len_secs) + '.pickle'

    with open(name, 'wb') as handle:
        pickle.dump(d, handle)

#ii_save_data()


shift = 0.25
samp_len_secs = 1

bc_save_data(shift, samp_len_secs)
#intensity_save_data_unclean(shift, samp_len_secs)
#intensity_save_data_clean(shift, samp_len_secs)