#import statements
import serial
import csv
import time 
import random
import numpy as np 
import pickle as pkl
import matplotlib.pyplot as plt 
import keyboard
from serial.serialutil import SerialException 
from joblib import load

#stuff to change each run 
participant_num = 2
dict_type = "scratch" 
#dict_type = "nonscratch" 

random.seed(participant_num) #participant number is also used as seed for randomizing the interaction order 

num_nonscratch = 7
num_scratch = 7 
if dict_type == "nonscratch": 
    interaction_dict = {1: 'hand waving', 2: 'keyboard typing', 3: 'texting/phone swiping', 
    4: 'writing', 5: 'table tapping', 6: 'air scratching', 7: 'clapping'}
    random_order = random.sample(range(1,1+num_nonscratch), num_nonscratch)
    print("Doing Non-Scratching Interactions")
elif dict_type == "scratch":
    interaction_dict = {8: 'top of hand/fingers', 9: 'forearm/wrist', 10: 'inside elbow', \
    11: 'neck', 12: 'head (on hair)', 13: 'behind the knees', 14: 'ankles'}
    random_order = random.sample(range(1+num_nonscratch,1+num_nonscratch+num_scratch), num_scratch)
    print("Doing Scratching Interactions")

#sampling/time variables 
num_secs = 30
samp_freq1 = 8000
samp_freq2 = 400

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

print("Default Starting Interaction number is 1")
interaction_start_num = int(input("Enter Starting Interaction Number: "))
#random_order = random_order[interaction_start_num-1:len(random_order)]
#num_interactions = len(random_order)
just_started = True
print(random_order)

#filenames
filenamestart = "/Users/akhil/Documents/Research/study/fig_dataset/" + str(participant_num) + "_"

#keyboard stuff
skip = 0 
stop = 0
restart = 0 
pause = 0
start_val = 0
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

interaction_count = 1
size_random_order = np.shape(random_order)[0]
ind = interaction_start_num

while ind <= size_random_order:
    i = random_order[ind-1]

    skip = 0
    stop = 0
    restart = 0
    pause = 0
    
    #print("Connected to Teensy ports")
    filename = filenamestart +  str(i)
    
    print()
    print()
    print("Interaction: " + interaction_dict[i])
    print("Count: ", ind)
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

    print("Starting in... ")            
    countdown = list(range(1,4))
    countdown.reverse()
    start = time.time()
    curr = time.time()
    
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

    for n in countdown:
        print(n)
        time.sleep(1)
        #ser.flushInput()
        #ser.read_all()
        #ser.reset_input_buffer()

    ser = serial.Serial(teensy_port, baud)
    if ser is None:
        raise RuntimeError('Serial Port is not found!')
    ser.reset_input_buffer()

    start = time.time()
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
                        print('Time Elapsed: ', curr-start, n1, n2)
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
    
    #open pickle file to write to it
    outfile = open(filename, 'wb')

    if skip == 1:
        print("Skipping interaction!")
        time.sleep(1)
        data_dict = {'t1' : [420], 'con' : [420], 't2' : [420], 'accx' : [420], 'accy' : [420], 'accz' : [420]}
        pkl.dump(data_dict, outfile, protocol=2)
        outfile.close()
        print("Interaction data collection complete", participant_num, i)
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
        pkl.dump(data_dict, outfile, protocol=2)
        outfile.close()
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

        if just_started or i % 3 == 0: #plot on 0th and every 3
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
                print("Interaction data collection complete", participant_num, i)
                print()
                ind+=1
        else:
            print("Stopped Interaction Early")
            print("Interaction data collection complete", participant_num, i)
            print()
            ind+=1
        
    start_val = 0
    keyboard.unhook_all_hotkeys()
    keyboard.unhook_all()


print("Data collection complete!")

