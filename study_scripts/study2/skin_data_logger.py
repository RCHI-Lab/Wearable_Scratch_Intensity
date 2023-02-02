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
eczema = 0 #if at CMU, eczema = 0. if at UPMC, eczema = 1
participant_num = 9



file_num = 1
num_secs = 8
if eczema == 0: 
    print("AT CMU")
    filenameheader = "/Volumes/T7/mod_dataset/healthy/Skin/"
elif eczema == 1:
    print("AT UPMC")
    filenameheader = "/Volumes/T7/mod_dataset/eczema/Skin/"
else: 
    print("Check eczema variable")

random.seed(participant_num) #participant number is also used as seed for randomizing the interaction order 
interaction_list = [1, 2, 3, 4, 5]
random.shuffle(interaction_list)

#sampling/time variables 
num_secs = 8
samp_freq1 = 5000
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


set_num = int(input("Enter Set Num: "))
print("Default Starting intensity number is 1")
interaction_start_num = int(input("Enter Starting Intensity Number: "))
just_started = True


#filenames
filenamestart = filenameheader + str(participant_num) + "_" + str(set_num) + "_"

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

i = interaction_start_num
while i <= 5:

    skip = 0
    stop = 0
    restart = 0
    pause = 0
    
    #print("Connected to Teensy ports")
    filename = filenamestart +  str(i)
    
    print()
    print()
    print("Intensity: " + str(i))
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
        i+=1
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
                i+=1
        else:
            print("Stopped Interaction Early")
            print("Interaction data collection complete", participant_num, i)
            print()
            i+=1
        
    start_val = 0
    keyboard.unhook_all_hotkeys()
    keyboard.unhook_all()


print("Data collection complete!")

