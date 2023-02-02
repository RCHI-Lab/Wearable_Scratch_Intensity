#import statements 
import os 
import sys
import time
from threading import Thread
from multiprocessing import Process
import binascii
import threading
import numpy as np 
import time
import csv 
from scipy import io
import copy
import pickle as pkl
import time
import serial
import matplotlib.pyplot as plt 
import keyboard
from serial.serialutil import SerialException 
import sys
sys.path.append('/Users/akhil/Documents/sensel-api/sensel-lib-wrappers/sensel-lib-python')
import sensel

#stuff to change each run 
participant_num = 25
file_num = 1

num_secs = 10
filenameheader = "/Users/akhil/Documents/Research/study/ii_dataset/"

######################
#sensel morph setup functions
######################
def initFrame(handle):
    error = sensel.setFrameContent(handle, sensel.FRAME_CONTENT_PRESSURE_MASK | sensel.FRAME_CONTENT_CONTACTS_MASK)
    (error, frame) = sensel.allocateFrameData(handle)
    error = sensel.startScanning(handle)
    return frame

def closeSensel(handle, frame):
    error = sensel.freeFrameData(handle, frame)
    error = sensel.stopScanning(handle)
    error = sensel.close(handle)

def scanFrames(handle, frame, info, start):
    new_data = []
    new_fm_data = []
    error = sensel.getFrame(handle, frame)
    read_time = (time.time() - start)*1e6
    #print(read_time)
    peak_forces = []
    peak_force = 0
    row = [np.nan]*6
    row[5] = read_time
    if frame.n_contacts > 0:
        #print()
        for n in range(frame.n_contacts):
            c = frame.contacts[n]
            #print(c.area, c.x_pos, c.y_pos, c.major_axis/c.minor_axis)
            if c.area >= 59 and c.area <= 300 and abs(c.major_axis/c.minor_axis)<= 2.0 and c.x_pos > 15.0 and c.x_pos < 225.0 and c.y_pos > 15.0 and c.y_pos < 110.0:
                if c.peak_force > peak_force:
                    peak_force = c.peak_force
                    row = [c.peak_force, c.total_force, c.area, c.x_pos, c.y_pos, read_time]
    #print(row)
    
    fa = np.ctypeslib.as_array(frame.force_array, shape=(info.num_rows,info.num_cols))
    #print(frame.force_array.contents)
    fa_new = copy.deepcopy(fa)
    #print(np.sum(fa))
        
    return row, fa_new

def openSensel():
    handle = None
    (error, device_list) = sensel.getDeviceList()
    if device_list.num_devices != 0:
        (error, handle) = sensel.openDeviceByID(device_list.devices[0].idx)
    return handle


#keyboard stuff
skip = 0 
stop = 0
restart = 0 
pause = 0
start_val = 0
just_started = True
def skip_func():
    global skip
    skip = 1
def stop_func(): 
    global stop
    stop = 1
def restart_func():
    global restart 
    restart = 1
def pause_func():
    global pause 
    pause = 1
def start_func():
    global start_val
    start_val = 1

interaction_dict = {
    1: 'low force, low speed', 2: 'low force, medium speed', 3: 'low force, high speed', 
    4: 'medium force, low speed', 5: 'medium force, medium speed', 6: 'medium force, high speed', 
    7: 'high force, low speed', 8: 'high force, medium speed', 9: 'high force, high speed',
    10: 'finger on tablet, low force, low speed', 11: 'finger on tablet, low force, medium speed', 12: 'finger on tablet, low force, high speed', 
    13: 'finger on tablet, medium force, low speed', 14: 'finger on tablet, medium force, medium speed', 15: 'finger on tablet, medium force, high speed', 
    16: 'finger on tablet, high force, low speed', 17: 'finger on tablet, high force, medium speed', 18: 'finger on tablet, high force, high speed'}

total_interaction_num = len(interaction_dict)
interaction_start_num = int(input("Enter Starting Interaction Number: "))



######################
#wearable function
######################

def wearable(participant_num, file_num, num_secs, other_start_time):
    global pause, restart, skip, stop, ind, just_started 
    #print("Started Wearable Thread")
    #serial initialization
    #teensy 4.0s
    teensy_port = "/dev/cu.usbmodem112127801" #teensy1
    #teensy_port = "/dev/cu.usbmodem112153601" #teensy2

    #teensy 3.0s
    #teensy_port = "/dev/cu.usbmodem100510501" #serial port of second Teensy
    #teensy_port = "/dev/cu.usbmodem107298201"
    #teensy_port = "/dev/cu.usbmodem94015901"
    #teensy_port = "/dev/cu.usbmodem94023501"
    baud = 115200
    samp_freq1 = 8000
    samp_freq2 = 400
    
    t = 0
    new_t = 0
    n1 = 0 
    n2 = 0
    c1 = []
    d1 = []
    c2 = []
    d2x = []
    d2y = []
    d2z = []
    buffer =  []

    ser = serial.Serial(teensy_port, baud)
    if ser is None:
        raise RuntimeError('Serial Port is not found!')
    ser.reset_input_buffer()

    start = time.time()
    print('Wearable Start Time:', start - other_start_time)

    while (n1 <= num_secs*samp_freq1 or n2 <= num_secs*samp_freq2) and (restart == 0 and skip == 0 and stop == 0 and pause == 0):
        buffer.append(ser.read())
        if len(buffer) >= 16:
            byte0 = int.from_bytes(buffer[0], byteorder='little')
            byte1 = int.from_bytes(buffer[1],byteorder='little')
            if byte0 == 255 and byte1 == 255:
                iden = int.from_bytes(buffer[2],byteorder='little')  
                if iden == 1:
                    val = int.from_bytes(buffer[3] + buffer[4],byteorder='little')
                    t1 = int.from_bytes(buffer[5] + buffer[6] + buffer[7] + buffer[8],byteorder='little')
                    if val <= 1023:  
                        c1.append(t1)
                        d1.append(val)
                        n1+=1
                    else:
                        print("Contact Mic BAD VAL")
                    if n1 % samp_freq1 == 0:
                        curr = time.time()
                        print('Time Elapsed: ', curr-start, n1)
                        print()
                elif iden == 3:
                    valx = int.from_bytes(buffer[3] + buffer[4],byteorder='little')
                    valy = int.from_bytes(buffer[5] + buffer[6] ,byteorder='little')
                    valz = int.from_bytes(buffer[7] + buffer[8] ,byteorder='little')
                    t2 = int.from_bytes(buffer[9] + buffer[10] + buffer[11] + buffer[12],byteorder='little')
                    signx = int.from_bytes(buffer[13],byteorder='little')
                    signy = int.from_bytes(buffer[14],byteorder='little')
                    signz = int.from_bytes(buffer[15],byteorder='little')
                    sx = 1
                    sy = 1
                    sz = 1
                    if signx == 0: 
                        sx = -1
                    if signy == 0: 
                        sy = -1
                    if signz == 0: 
                        sz = -1
                    if abs(valx)/100 < 21 and abs(valy)/100 < 21 and abs(valz)/100 < 21:
                        c2.append(t2)
                        d2x.append(valx*sx) 
                        d2y.append(valy*sy)
                        d2z.append(valz*sz)
                        n2+=1
                    else:
                        print("Acc BAD VALUE")
            buffer.pop(0)
    ser.close()

    if skip == 1:
        print("Skipping interaction!")
        time.sleep(1)
        data_dict = {'t1' : [420], 'con' : [420], 't2' : [420], 'accx' : [420], 'accy' : [420], 'accz' : [420]}
        pkl.dump(data_dict, outfile, protocol=2)
        outfile.close()
        ind+=1
    elif restart == 1: 
        print("Restarting interaction!")
        time.sleep(1)
    elif pause == 1: 
        pause = 0
        print()
        print("Press p to continue!")
        print()
        while pause == 0: 
            time.sleep(1)
        pause = 0
    else:
        data_dict = {'t1' : c1, 'con' : d1, 't2' : c2, 'accx' : d2x, 'accy' : d2y, 'accz' : d2z}
        filenamestart = filenameheader + 'w_' + str(participant_num) + "_"
        filename = filenamestart +  str(file_num)
        outfile1 = open(filename, 'wb')
        pkl.dump(data_dict, outfile1, protocol=2)
        outfile1.close()
        print('Num of Samples', n1, n2)
        curr = time.time()
        py_time = curr - start
        print('Time Elapsed (Python): ', py_time)
        t1 = (c1[n1-1]-c1[0])/1e6
        t2 = (c2[n2-1]-c2[0])/1e6
        print('Time Elapsed (Teensy): ', t1, t2) 

        con_min = min(d1)
        con_max = max(d1)
        acc_min = min(d2z)/100
        acc_max = max(d2z)/100
        print("Cont Mic: ", con_min, con_max)
        print("Acc: ", acc_min , acc_max)
        
        '''
        if just_started or ind % 3 == 0: #plot on 0th and every 3
            # Create two subplots and unpack the output array immediately
            just_started = False
            plt.subplot(4, 1, 1)
            plt.plot(c1, d1)

            plt.subplot(4, 1, 2)
            plt.plot(c2,d2x)

            plt.subplot(4, 1, 3)
            plt.plot(c2,d2y)

            plt.subplot(4, 1, 4)
            plt.plot(c2,d2z)

            plt.show()
        '''
        #check to make sure data isn't messed up 
        teensy_sec_tol = 0.1
        python_sec_tol = 0.2

        if stop == 0:
            if (abs(t1) > (num_secs + teensy_sec_tol) or abs(t1) < (num_secs - teensy_sec_tol)) or (abs(t2) > (num_secs + teensy_sec_tol) or abs(t2) < (num_secs - teensy_sec_tol)):
                print()
                print("redoing sample cause of teensy timing issue!")
                print()
            elif py_time > (num_secs + python_sec_tol): 
                print()
                print("redoing sample cause of python timing issue!")
                print()
            elif con_min < 0 or con_max > 1023 or con_min == con_max:
                print()
                print('contact mic issues')
                print()
            elif acc_min < -21 or acc_max > 21 or acc_min == acc_max:
                print()
                print('acc issues')
                print()
            else:
                #print("Interaction data collection complete", participant_num, i)
                print()
                ind+=1
        else:
            print("Stopped Interaction Early")
            #print("Interaction data collection complete", participant_num, i)
            print()
            ind+=1
        
        


#main stuff
size_interactions = len(interaction_dict)
ind = interaction_start_num
while ind <= size_interactions:
    file_num = ind
    skip = 0
    stop = 0
    restart = 0
    pause = 0

    print()
    print()
    print("Interaction " + str(file_num) + ": " + interaction_dict[ind])
    print("Press s if you would like to skip this interaction, space if you would like stop this interaction, r to restart this interaction, or p to pause at any time")
    print()
    keyboard.add_hotkey('shift', start_func)
    keyboard.add_hotkey('s', skip_func)
    keyboard.add_hotkey('p', pause_func)
    print("Press shift when you're ready to start!")
    while start_val != 1: 
        time.sleep(1)

    keyboard.remove_hotkey('shift')
    keyboard.add_hotkey('space', stop_func)
    keyboard.add_hotkey('r', restart_func)


    start_time = time.time()
    handle = openSensel()
    if handle == None:
        print("Sensel not connecting.")
    else:
        #set scan detail 
        detail = 1  # High(0), Medium(1), Low(2)
        error = sensel.setScanDetail(handle, detail)
        print("Setting Scan Detail Error:", error)

        #set frame rate 
        rate = 150
        error = sensel.setMaxFrameRate(handle, rate)
        print("Setting Frame Rate Error:", error)

        #get frame rate to verify it worked
        (error, rate_read) = sensel.getFrameRate(handle)
        print("Max Frame Rate set to :", rate_read, "Hz") 

        #set contact bitmask to get peak force also 
        error = sensel.setContactsMask(handle, sensel.CONTACT_MASK_PEAK | sensel.CONTACT_MASK_ELLIPSE)
        print("Setting Contact Bitmask Error:", error)

        #set sensel min force 
        force = 40 #default is 160 (20 g). divide by 8 to get into grams. 40 is the lowest possible 
        error = sensel.setContactsMinForce(handle, force)
        print("Setting Min Force Error:", error)

        #buffer control 
        size = 0
        error = sensel.setBufferControl(handle,size)
        print("Setting Buffer Size Error:", error)

        #verify buffer control worked
        (error, size_read) = sensel.getBufferControl(handle)
        print("Buffer Size set to:", size_read)

        (error, info) = sensel.getSensorInfo(handle)

    print()
    print("Starting in... ")            
    countdown = list(range(1,4))
    countdown.reverse()
    for n in countdown:
        print(n)
        time.sleep(1)
    print('Starting Data Collection')
    time.sleep(1) #wait an extra second so that initial data is good

    start_time = time.time()

    p1 = threading.Thread(target = wearable, args = (participant_num, file_num, num_secs, start_time))
    p1.start()
    #print("Started Sensel Morph")
    full_data = []
    force_map = []

    frame = initFrame(handle)
    print("Sensel Start Time:", time.time() -start_time)
    start = time.time()
    dt = time.time() - start
    while(dt <= num_secs) and (restart == 0 and skip == 0 and stop == 0 and pause == 0):
        error = sensel.readSensor(handle)
        (error, num_frames) = sensel.getNumAvailableFrames(handle)
        if num_frames > 0:
            new_data, new_fm_data = scanFrames(handle, frame, info, start)
            #print(new_data)
            full_data.append(new_data)
            force_map.append(new_fm_data)
        dt = time.time() - start
    print("Sensel End Time: ", time.time()-start_time )
    closeSensel(handle, frame)
    p1.join()
    full_data = np.array(full_data)
    #print("Data Shape:", np.shape(full_data))
    #print("FM Data Shape:", np.shape(force_map))


    if skip == 1: 
        data_dict = {'peakforce' : [420], 'totalforce' : [420], \
        'area' : [420], 'xpos' : [420], 'ypos' : [420], \
        'timestamp' :[420]}
        outfile = open(filename, 'wb')
        pkl.dump(data_dict, outfile, protocol=2)
        outfile.close()
    else: 
        #write to pickle
        filenamestart = filenameheader + 's_' + str(participant_num) + "_"
        filename = filenamestart + str(file_num)
        data_dict = {'peakforce' : full_data[:,0], 'totalforce' : full_data[:,1], \
            'area' : full_data[:,2], 'xpos' : full_data[:,3], 'ypos' : full_data[:,4], \
            'timestamp' :full_data[:,5]}
        outfile = open(filename, 'wb')
        pkl.dump(data_dict, outfile, protocol=2)
        outfile.close()
        print("Exported to pkl file")
        print("File Num: ", file_num)
    start_val = 0
    just_started = False
    keyboard.unhook_all_hotkeys()
    keyboard.unhook_all()
    #heatmap stuff 
    '''
    force_map = {'fm': force_map}
    filenamestart = "/Users/akhil/Documents/Research/final/senseldataset/" + 'fm_' + str(participant_num) + "_"
    filename = filenamestart + str(file_num) + ".mat"
    io.savemat(filename,force_map)
    '''

