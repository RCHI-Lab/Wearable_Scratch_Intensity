import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random
import pickle 
from scipy.interpolate import interp1d
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import itertools
from itertools import product
from joblib import Parallel, delayed 
from scipy.signal import savgol_filter, find_peaks
from scipy import signal
from os.path import exists 


def get_bc_data(num_secs, shift, samp_freq1, samp_freq2, samp_len_secs):
    #import all data 
    num_participants = 24
    num_participants_study = num_participants
    num_files = 14 

    excluded_participants = np.array([1,4,6,7]) - 1
    participant_nums = np.arange(1,num_participants+1)
    participant_nums = np.delete(participant_nums, excluded_participants)
    file_nums = np.arange(1, num_files+1)
    combs = list(product(participant_nums, file_nums))

    print('Getting Data')
    fh = '/home/akhil/Documents/research/final_code/data/bc_dataset/'
    out = Parallel(n_jobs = -1, backend = 'threading')(delayed(process_wearable_data)(num_secs, shift, p, f, fh, samp_freq1, samp_freq2, samp_len_secs) for p,f in combs)
    out = np.array(out, dtype = object)
    wearable_data = np.asarray(np.stack(np.reshape(out[:,0,:], (-1)), axis = 0)).astype(np.float32)
    labels = np.asarray(np.reshape(out[:,1,:], (-1))).astype(np.float32)
    loso_labels = np.asarray(np.reshape(out[:,2,:], (-1))).astype(np.float32)
    cat_labels = np.asarray(np.reshape(out[:,3,:], (-1))).astype(np.float32)
    return wearable_data, labels, loso_labels, cat_labels

def get_intensity_data_unclean(num_secs, shift, samp_freq1, samp_freq2, samp_len_secs):
    num_participants = 24
    num_participants_study = num_participants
    num_files = 9

    excluded_participants = np.array([1,4,6,7]) - 1
    participant_nums = np.arange(1,num_participants+1)
    participant_nums = np.delete(participant_nums, excluded_participants)
    file_nums = np.arange(1, num_files+1)
    combs = list(product(participant_nums, file_nums))
    print('Getting Data')
    all_wearable_data = []
    all_wearable_data_vel = []
    all_velocity_labels = []
    all_force_labels = []
    all_power_labels = []
    all_loso_labels = []
    all_loso_labels_vel = []
    for p,f in combs:
        fileheader = '/home/akhil/Documents/research/final_code/data/ii_dataset/'
        fh = fileheader + 'w_'
        wearable_data,_,loso_labels,_ = process_wearable_data(num_secs, shift, p, f, fh, samp_freq1, samp_freq2, samp_len_secs)
        fh = fileheader + 's_'        
        velocitylabels, forcelabels, powerlabels = process_sensel_data(num_secs, shift, p, f, fh, samp_len_secs) 
        
        all_wearable_data.extend(wearable_data)
        all_force_labels.extend(forcelabels)
        all_velocity_labels.extend(velocitylabels)
        all_power_labels.extend(powerlabels) 
        all_loso_labels.extend(loso_labels)
    
    return all_wearable_data, all_force_labels, all_velocity_labels, all_power_labels, all_loso_labels


def get_intensity_data_clean(num_secs, shift, samp_freq1, samp_freq2, samp_len_secs):
    num_participants = 24
    num_participants_study = num_participants
    num_files = 9 

    excluded_participants = np.array([1,4,6,7]) - 1
    participant_nums = np.arange(1,num_participants+1)
    participant_nums = np.delete(participant_nums, excluded_participants)
    file_nums = np.arange(1, num_files+1)
    combs = list(product(participant_nums, file_nums))
    print('Getting Data')
    all_wearable_data = []
    all_wearable_data_vel = []
    all_velocity_labels = []
    all_force_labels = []
    all_power_labels = []
    all_loso_labels = []
    all_loso_labels_vel = []
    for p,f in combs:
        fileheader = '/home/akhil/Documents/research/final_code/data/ii_dataset/'
        fh = fileheader + 'w_'
        wearable_data,_,loso_labels,_ = process_wearable_data(num_secs, shift, p, f, fh, samp_freq1, samp_freq2, samp_len_secs)
        fh = fileheader + 's_'        
        velocitylabels, forcelabels, powerlabels = process_sensel_data_cleaner(num_secs, shift, p, f, fh, samp_len_secs) 
        
        #remove nan data 
        inds = np.argwhere(np.isnan(powerlabels))
        wearable_data = np.delete(wearable_data, inds, axis = 0)
        powerlabels = np.delete(powerlabels, inds, axis = 0) 
        velocitylabels = np.delete(velocitylabels, inds, axis = 0) 
        forcelabels = np.delete(forcelabels, inds, axis = 0) 
        loso_labels = np.delete(loso_labels, inds, axis = 0) 
        
        #remove data greater than 600 mW
        bound = 600
        inds = np.argwhere(powerlabels > bound)
        wearable_data = np.delete(wearable_data, inds, axis = 0)
        powerlabels = np.delete(powerlabels, inds, axis = 0) 
        velocitylabels = np.delete(velocitylabels, inds, axis = 0) 
        forcelabels = np.delete(forcelabels, inds, axis = 0) 
        loso_labels = np.delete(loso_labels, inds, axis = 0) 
       
        all_wearable_data.extend(wearable_data)
        all_force_labels.extend(forcelabels)
        all_velocity_labels.extend(velocitylabels)
        all_power_labels.extend(powerlabels) 
        all_loso_labels.extend(loso_labels)

    return all_wearable_data, all_force_labels, all_velocity_labels, all_power_labels, all_loso_labels



def process_wearable_data(num_secs, shift, participant_num, file_num, fh, samp_freq1, samp_freq2, samp_len_secs):
    file = fh + str(participant_num) + '_' + str(file_num)
    data = pd.read_pickle(file)

    step1 = 1/samp_freq1
    step2 = 1/samp_freq2
    sample_len1 = int(samp_freq1*samp_len_secs)
    sample_len2 = int(samp_freq2*samp_len_secs)
    
    t1 = np.array(data['t1'])
    t2 = np.array(data['t2'])
    con = np.array(data['con'])
    accx = np.array(data['accx'])/100
    accy = np.array(data['accy'])/100
    accz = np.array(data['accz'])/100
    
    skipped_bool = t1[0] == 420 and t2[0] == 420
    if skipped_bool:
        return 420
    
    t1 = (t1 - t1[0])/1e6 #zero and convert to seconds
    t2 = (t2 - t2[0])/1e6 #zero and convert to seconds
    

    #interpolate con mic data 
    tcon = np.arange(0, num_secs, step1)
    f = interp1d(t1, con)
    con = f(tcon)

    #interpolate acc data 
    tacc = np.arange(0, num_secs, step2)
    #f = interp1d(t2, accx)
    #accx = f(tacc)
    #f = interp1d(t2, accy)
    #accy = f(tacc)
    f = interp1d(t2, accz)
    accz = f(tacc)

    wearable_data = []
    labels = []
    loso_labels = []
    file_labels = []
    if file_num >=8:
        label = 1
    else:
        label = 0 
    
    #num_data = int((num_secs-1)/shift + 1)
    num_data = int((num_secs-samp_len_secs)/shift/samp_len_secs + 1)
    #index_range = np.linspace(0, num_secs-1.0, num_data, endpoint = True)
    index_range = np.linspace(0, num_secs/samp_len_secs-samp_len_secs, num_data, endpoint = True)

    for ind in index_range:
        data = []
        con_ind = int(sample_len1*ind)
        acc_ind = int(sample_len2*ind)
        con_crop = con[con_ind:con_ind+sample_len1]
        accz_crop = accz[acc_ind:acc_ind+sample_len2]
        data = np.array(con_crop)
        data = np.append(data, np.array(accz_crop))
    
        if np.shape(con_crop)[0] != sample_len1:
            print(np.shape(con_crop), sample_len1, num_data, ind)
        if np.shape(accz_crop)[0] != sample_len2:
            print(np.shape(accz_crop), sample_len2)
        
        wearable_data.append(np.array(data))
        labels.append(label)
        loso_labels.append(participant_num)
        file_labels.append(file_num)
    #print(np.shape(wearable_data), np.shape(labels))
    return wearable_data, labels, loso_labels, file_labels

def process_sensel_data(num_secs, shift, participant_num, file_num, fh, samp_len_secs):
    #print()
    interaction_dict = {
    1: 'low force, low speed', 2: 'low force, medium speed', 3: 'low force, high speed', 
    4: 'medium force, low speed', 5: 'medium force, medium speed', 6: 'medium force, high speed', 
    7: 'high force, low speed', 8: 'high force, medium speed', 9: 'high force, high speed',
    10: 'finger on tablet, low force, low speed', 11: 'finger on tablet, low force, medium speed', 12: 'finger on tablet, low force, high speed', 
    13: 'finger on tablet, medium force, low speed', 14: 'finger on tablet, medium force, medium speed', 15: 'finger on tablet, medium force, high speed', 
    16: 'finger on tablet, high force, low speed', 17: 'finger on tablet, high force, medium speed', 18: 'finger on tablet, high force, high speed'}
    
    #print("Participant Num:", participant_num, "   File Num:", file_num, "   Interaction: ", interaction_dict[file_num])
    file = fh + str(participant_num) + '_' + str(file_num)
    data = pd.read_pickle(file)


    dts = data['timestamp']
    xpos = data['xpos']
    ypos = data['ypos']
    totalforces = data['totalforce']
    totalareas = data['area']

    #interpolate x and y position data 
    ok = ~np.isnan(ypos)
    xp = ok.ravel().nonzero()[0]
    fp = ypos[~np.isnan(ypos)]
    x  = np.isnan(ypos).ravel().nonzero()[0]
    ypos_in = np.copy(ypos)
    ypos_in[np.isnan(ypos)] = np.interp(x, xp, fp)

    ok = ~np.isnan(xpos)
    xp = ok.ravel().nonzero()[0]
    fp = xpos[~np.isnan(xpos)]
    x  = np.isnan(xpos).ravel().nonzero()[0]
    xpos_in = np.copy(xpos)
    xpos_in[np.isnan(xpos)] = np.interp(x, xp, fp)


    yhat = savgol_filter(ypos_in, 31, 5)
    thresh = np.mean([max(yhat), min(yhat)])
    dist = 20
    peaks, _ = find_peaks(yhat, height=thresh, distance = dist)
    valleys, _ = find_peaks(-yhat, height=-thresh,distance = dist)


    porvs = []
    times = []
    pvinds = []
    times.extend(dts[peaks].flatten().tolist())
    porvs.extend(np.zeros(np.shape(dts[peaks])[0]))
    pvinds.extend(peaks)
    times.extend(dts[valleys].flatten().tolist())
    porvs.extend(np.ones(np.shape(dts[valleys])[0]))
    pvinds.extend(valleys)

    xs = []
    xs.extend(xpos_in[peaks].flatten().tolist())
    xs.extend(xpos_in[valleys].flatten().tolist())

    ys = []
    ys.extend(ypos_in[peaks].flatten().tolist())
    ys.extend(ypos_in[valleys].flatten().tolist())

    ind = np.argsort(times)
    times = np.array(times)[ind]
    xs = np.array(xs)[ind]
    ys = np.array(ys)[ind]
    porvs = np.array(porvs)[ind]
    pvinds = np.array(pvinds)[ind]
    
    
    #num_data = int((num_secs-1)/shift + 1)
    #sec_range = (np.linspace(0, num_secs-1, num = num_data)*1e6).astype(int)
    num_data = int((num_secs-samp_len_secs)/shift/samp_len_secs + 1)
    sec_range = (np.linspace(0, num_secs/samp_len_secs-samp_len_secs, num_data)*1e6).astype(int)

    
    vel_agg = []
    all_agg = []
    num = 0 
    for s in sec_range:
        s = round(s)
        s2 = round(s+1e6)
        time = times[((times >= s) & (times < s2))]
        porv = porvs[((times >= s) & (times < s2))]
        pvind = pvinds[((times >= s) & (times < s2))]
        x = xs[((times >= s) & (times < s2))]
        y = ys[((times >= s) & (times < s2))]
        velocities = [] 
        forces = []
        pressures = []
        agg = [0]*8
        
        for n in np.arange(0, np.shape(time)[0]-1):
            currporv = porv[n]
            nextporv = porv[n+1]
            if currporv != nextporv:
                currx = x[n]
                curry = y[n]
                currt = time[n]
                nextx = x[n+1]
                nexty= y[n+1]
                nextt = time[n+1]
                startind = pvind[n]
                endind = pvind[n+1]
                dt = (nextt-currt)/1e6
                dist = np.sqrt((nexty-curry)**2 + (nextx-currx)**2)
                velocities.append(dist/dt)
                    
             
        #print(velocities)
        avg_vel = np.nanmean(velocities)
        med_vel = np.nanmedian(velocities)
        vel_agg.append([avg_vel, med_vel])
        agg[4] = np.nanmean(totalforces)
        agg[5] = np.nanmean(pressures)
        all_agg.append(agg)
        
    combined = np.hstack((all_agg,vel_agg))

    #print()
    #print("Intensity Level:", intensity)
 
    df = pd.DataFrame(combined, columns = ['Average Peak Force (g)', 'Average Peak Pressure (g/mm^2)', 
                                           'Median Peak Force (g)', 'Median Peak Pressure (g/mm^2)', 
                                           'Average Total Force (g)', 'Average Total Pressure (g/mm^2)', 
                                           'Median Total Force (g)', 'Median Total Pressure (g/mm^2)',
                                           'Average Velocity (mm/s)', 'Median Velocity (mm/s)'])
    df = df[['Average Total Force (g)', 'Average Velocity (mm/s)']]

    #display(df)
    velocitylabels = []
    forcelabels = []
    powerlabels = []
    for vals in combined:
        avg_pressure = vals[5]*0.0098*(1000**2) #pressure in Pa
        avg_total_force = vals[4]*0.0098 #converting to N 
        avg_vel = vals[8]/1000 #converting to m/s
        power = avg_total_force*avg_vel*1000 #in units of mW now 
        vel = avg_vel*1000 #mm/s
        force = avg_total_force #N
        forcelabels.append(force) 
        velocitylabels.append(vel)
        powerlabels.append(power)
    #print(np.nanmean(forcelabels), np.nanmean(velocitylabels), np.nanmean(power))
    return velocitylabels, forcelabels, powerlabels 


def process_sensel_data_cleaner(num_secs, shift, participant_num, file_num, fh, samp_len_secs):
    #print()
    interaction_dict = {
    1: 'low force, low speed', 2: 'low force, medium speed', 3: 'low force, high speed', 
    4: 'medium force, low speed', 5: 'medium force, medium speed', 6: 'medium force, high speed', 
    7: 'high force, low speed', 8: 'high force, medium speed', 9: 'high force, high speed',
    10: 'finger on tablet, low force, low speed', 11: 'finger on tablet, low force, medium speed', 12: 'finger on tablet, low force, high speed', 
    13: 'finger on tablet, medium force, low speed', 14: 'finger on tablet, medium force, medium speed', 15: 'finger on tablet, medium force, high speed', 
    16: 'finger on tablet, high force, low speed', 17: 'finger on tablet, high force, medium speed', 18: 'finger on tablet, high force, high speed'}
    
    #print("Participant Num:", participant_num, "   File Num:", file_num, "   Interaction: ", interaction_dict[file_num])
    file = fh + str(participant_num) + '_' + str(file_num)
    data = pd.read_pickle(file)


    dts = data['timestamp']
    xpos = data['xpos']
    ypos = data['ypos']
    totalforces = data['totalforce']
    totalareas = data['area']
    
    
    #interpolate x and y position data 
    ok = ~np.isnan(ypos)
    xp = ok.ravel().nonzero()[0]
    fp = ypos[~np.isnan(ypos)]
    x  = np.isnan(ypos).ravel().nonzero()[0]
    ypos_in = np.copy(ypos)
    ypos_in[np.isnan(ypos)] = np.interp(x, xp, fp)

    ok = ~np.isnan(xpos)
    xp = ok.ravel().nonzero()[0]
    fp = xpos[~np.isnan(xpos)]
    x  = np.isnan(xpos).ravel().nonzero()[0]
    xpos_in = np.copy(xpos)
    xpos_in[np.isnan(xpos)] = np.interp(x, xp, fp)

    yhat = savgol_filter(ypos_in, 31, 5)
    #plt.plot(dts[plotting_start_ind:plotting_end_ind], yhat[plotting_start_ind:plotting_end_ind]) #plot interpolated and smoothed signal 
    #plt.title('Smoothed YPosition vs Time with Mins/Maxs')
    thresh = np.mean([max(yhat), min(yhat)])
    dist = 20
    peaks, _ = find_peaks(yhat, height=thresh, distance = dist)
    valleys, _ = find_peaks(-yhat, height=-thresh,distance = dist)

    porvs = []
    times = []
    pvinds = []
    times.extend(dts[peaks].flatten().tolist())
    porvs.extend(np.zeros(np.shape(dts[peaks])[0]))
    pvinds.extend(peaks)
    times.extend(dts[valleys].flatten().tolist())
    porvs.extend(np.ones(np.shape(dts[valleys])[0]))
    pvinds.extend(valleys)

    xs = []
    xs.extend(xpos_in[peaks].flatten().tolist())
    xs.extend(xpos_in[valleys].flatten().tolist())

    ys = []
    ys.extend(ypos_in[peaks].flatten().tolist())
    ys.extend(ypos_in[valleys].flatten().tolist())

    ind = np.argsort(times)
    times = np.array(times)[ind]
    xs = np.array(xs)[ind]
    ys = np.array(ys)[ind]
    porvs = np.array(porvs)[ind]
    pvinds = np.array(pvinds)[ind]
    
    num_data = int((num_secs-samp_len_secs)/shift/samp_len_secs + 1)
    sec_range = (np.linspace(0, num_secs/samp_len_secs-samp_len_secs, num_data)*1e6).astype(int)

    
    vel_agg = []
    all_agg = []
    num = 0 
    plotting_num = 0
    for s in sec_range:
        s = round(s)
        s2 = round(s+1e6)
        time = times[((times >= s) & (times < s2))]
        porv = porvs[((times >= s) & (times < s2))]
        pvind = pvinds[((times >= s) & (times < s2))]
        x = xs[((times >= s) & (times < s2))]
        y = ys[((times >= s) & (times < s2))]
        
    
        x_pos_range = xpos[((dts >= s) & (dts < s2))]
        y_pos_range = ypos[((dts >= s) & (dts < s2))]
        delta_x = np.diff(x_pos_range)
        delta_y = np.diff(y_pos_range)
        cond3_thresh = 5 #mm 
        cond3 = (np.sum(np.abs(delta_x) > cond3_thresh) + np.sum(np.abs(delta_y) > cond3_thresh)) > 0
        velocities = [] 
        forces = []
        pressures = []
        agg = [0]*8

        cond1 = False
        cond2 = False
        cond4 = False
        cond1 = np.shape(porv)[0] < 2
        
        if cond1 == True or cond3 == True:
            velocities = [np.nan]
            #forces = [np.nan]
            pressures = [np.nan]
        else: 
            for n in np.arange(0, np.shape(time)[0]-1):
                currporv = porv[n]
                nextporv = porv[n+1]
                if currporv != nextporv:
                    currx = x[n]
                    curry = y[n]
                    currt = time[n]
                    nextx = x[n+1]
                    nexty= y[n+1]
                    nextt = time[n+1]
                    startind = pvind[n]
                    endind = pvind[n+1]
                    dt = (nextt-currt)/1e6
                    dist = np.sqrt((nexty-curry)**2 + (nextx-currx)**2)
                    y_array = ypos[startind:endind]
                    perc_nan = np.sum(np.isnan(y_array))/np.shape(y_array)[0]
                    if perc_nan > 0.50:
                        v = np.nan
                        cond2 = True
                    else:
                        v = dist/dt
                        total_area_array = totalareas[startind:endind]
                        avg_pressure = 0 
                    velocities.append(v)
                else: #this section removes data where there are two peaks or two valleys in a row
                    #print("Two peaks/valleys in a row")
                    velocities = [np.nan]
                    #forces = [np.nan]
                    pressures = [np.nan]
                    cond4 = True 
                    break
    
        
        #print(velocities)
        avg_vel = np.nanmean(velocities)
        med_vel = np.nanmedian(velocities)
        vel_agg.append([avg_vel, med_vel])
        agg[4] = np.nanmean(totalforces)
        agg[5] = np.nanmean(pressures)
        all_agg.append(agg)
            
        plotting_num+=1
        if False :#and cond1 == True or cond3 == True:#plotting_num % 30 == 0:#err > 300 and label < 600: #np.isnan(label) and False: #err > 300 and label < 600 or cond1 or cond2: 
            print()
            print()
            print()
            if cond1 == True:
                print('cond 1')
            if cond3 == True:
                print('cond 3')
            if cond4 == True:
                print('cond 4')
            print('huge error:', err, "   Label:", label, '     Pred:', pred)
            print('force label: ', agg[4]*0.0098, '     velocity label: ', avg_vel)
           # print(pvind)
           # print(time)
           # print(y)
    
            t = dts[((dts >= s) & (dts < s2))]
            y_all = ypos[((dts >= s) & (dts < s2))]
            forces = totalforces[((dts >= s) & (dts < s2))]
            plt.plot(t,y_all)
            plt.title("Sensel Morph Contact Y Position")
            
            #v = y[pvind]
            
            plt.scatter(time, y)
            plt.show()
            
            
            ###############################################################################
            
            plt.plot(t, forces*0.0098)
            plt.title("Sensel Morph Force")
            plt.show()
        
            '''
            plt.plot(wd_1[0:8000])
            plt.title("Contact Mic")
            plt.show()
            plt.plot(wd_1[8000:])
            plt.title("Accelerometer")
            plt.show()
            '''

    combined = np.hstack((all_agg,vel_agg))

    #print()
    #print("Intensity Level:", intensity)
 
    df = pd.DataFrame(combined, columns = ['Average Peak Force (g)', 'Average Peak Pressure (g/mm^2)', 
                                           'Median Peak Force (g)', 'Median Peak Pressure (g/mm^2)', 
                                           'Average Total Force (g)', 'Average Total Pressure (g/mm^2)', 
                                           'Median Total Force (g)', 'Median Total Pressure (g/mm^2)',
                                           'Average Velocity (mm/s)', 'Median Velocity (mm/s)'])
    df = df[['Average Total Force (g)', 'Average Velocity (mm/s)']]

    #display(df)
    velocitylabels = []
    forcelabels = []
    powerlabels = []
    for vals in combined:
        avg_pressure = vals[5]*0.0098*(1000**2) #pressure in Pa
        avg_total_force = vals[4]*0.0098 #converting to N 
        avg_vel = vals[8]/1000 #converting to m/s
        power = avg_total_force*avg_vel*1000 #in units of mW now 
        vel = avg_vel*1000 #mm/s
        force = avg_total_force #N
        forcelabels.append(force) 
        velocitylabels.append(vel)
        powerlabels.append(power)
    return velocitylabels, forcelabels, powerlabels 


def standardize(X_train, X_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def normalize(X_train, X_test):
    normalizer = MinMaxScaler().fit(X_train)
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train, X_test, normalizer 

def apply_fft(X_train, X_test, con_len, acc_len, sample_len1, sample_len2):
    X_train_con = X_train[:,0:sample_len1]
    X_train_acc = X_train[:,sample_len1:sample_len1+sample_len2]
    X_test_con = X_test[:,0:sample_len1]
    X_test_acc = X_test[:,sample_len1:sample_len1+sample_len2]
    
    fft_result = fft(X_train_con)
    yf = 2.0/sample_len1 * np.abs(fft_result[:,0:sample_len1//2])
    X_train_dr1 = yf[:,0:con_len]
    
    fft_result = fft(X_test_con)
    yf = 2.0/sample_len1 * np.abs(fft_result[:,0:sample_len1//2])
    X_test_dr1 = yf[:,0:con_len]
    
    fft_result = fft(X_train_acc)
    yf = 2.0/sample_len2 * np.abs(fft_result[:,0:sample_len2//2])
    X_train_dr2 = yf[:,0:acc_len]
    
    fft_result = fft(X_test_acc)
    yf = 2.0/sample_len2 * np.abs(fft_result[:,0:sample_len2//2])
    X_test_dr2 = yf[:,0:acc_len]

    X_train_dr = np.concatenate((X_train_dr1, X_train_dr2), axis = 1)
    X_test_dr = np.concatenate((X_test_dr1, X_test_dr2), axis = 1)

    return X_train_dr, X_test_dr





