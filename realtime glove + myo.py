import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import myo
from threading import Lock, Thread
from libRMS import * #library created specifically for EMG signal processing
import joblib

# --------------------------------------------------------------------------------------
# FINGER ANGLE GROUND TRUTH CODE
# --------------------------------------------------------------------------------------
ser = serial.Serial('com4',baudrate=2000000)
data = []
start_time = time.time()

seconds = 90
collectt=True


#load the prediction model
model1 = joblib.load('SVM_naufal.pkl')


def flex():
    
    global collectt
    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time-start_time
        
        arduino_data = ser.readline().decode('ascii', errors = 'ignore')
        arduino_data = arduino_data.strip().split(" ")
        # data.append(arduino_data)
        
        if elapsed_time >= 1:
            break
    time.sleep(3)
    
    start_time = time.time()
    print('=========S T A  R T==========')
    while collectt:
        current_time = time.time()
        elapsed_time = current_time-start_time
        
        arduino_data = ser.readline().decode('ascii', errors = 'ignore')
        arduino_data = arduino_data.strip().split(" ")
        data.append(arduino_data)
        t = int(elapsed_time)
        print(t, end='\r')
        if t==0 or t==15 or t==30 or t==45 or t==60 or t==75:
            print('      =======> FIST <========', end='\r' )
        elif t==5 or t==20 or t==35 or t==50 or t==65 or t==80:
            print('      =========> RELEASE <=======', end='\r')
        elif t==10 or t==25 or t==40 or t==55 or t==70 or t==85:
            print('      =========> REST <======', end='\r')

        if elapsed_time >= seconds:
            collectt=False
            break
    data2=pd.DataFrame(data)
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    data2.to_csv('ground_truth_finger_angle.csv', index=False) #<------------save the groundtruth finger angle data----------
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # print(len(data2))
    print("========While Loop Stop=========")

# -----------------------------------------------------------------------------------
# EMG SIGNAL STREAMING CODE + PROCESSING (MYO ARMBAND SENSOR)
# -----------------------------------------------------------------------------------

class Plot(object):
    def __init__(self, listener):
        self.n = listener.n
        self.listener = listener
        self.fig = plt.figure()
        self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
        [(ax.set_ylim([-100, 100])) for ax in self.axes]
        self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
        plt.ion()

    def main(self):
        pred = []
        while collectt:
            
            plt.pause(1/5) 
            emg_data = self.listener.get_emg_data()
            emg_data = np.array([x[1] for x in emg_data])
            emgX = pd.DataFrame(emg_data)

            data = emgX.shape
            # print(oi)

             #we use a window size of 40, the size of this EMG tuple code when initially sending data is less than 40 which causes an error, 
             #then "if" here is to wait for the tuple size to become 40, then the EMG data streaming program can run
            if data[0] > 39:
                
                emg_data = np.array(emgX).reshape(-1,8)
                save_emg = pd.DataFrame(emg_data)
                save_emg.columns = ['ch1', 'ch2', 'ch3','ch4','ch5','ch6','ch7','ch8']
                datafitur = prep(save_emg)
                realtime = model1.predict(datafitur)
               
                print(realtime)
                pred.append(realtime)
               
        pred1 = pd.DataFrame(np.array(pred).reshape(-1,5)) #5 is five fingers
        pred1.to_csv('predict_finger_angle_SVM.csv')#save the prediction results      
            
        
            
            
        

    def main(self):
        while collectt:
            self.update_plot()
            plt.pause(1.0 / 5) 
            #don't forget to set plt.pause, here we use a 200 ms sliding window, so that in 1 second there are 5 predictions

def main():
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(40) #The amount of EMG data sent depends on the window size
    with hub.run_in_background(listener.on_event):
        Plot(listener).main()


# -------------------------------------------------------------------------------------
# RUN THE FINGER ANGLE PREDICTION CODE BASED ON THE EMG SIGNAL THAT HAS BEEN PROCESSED + THE FINGER ANGLE PRECEDING CODE AS GROUNDTRUTH
# -------------------------------------------------------------------------------------

def get_data():
    p1 =Thread(target=flex)
    p2 =Thread(target=main)

    p1.start()
    p2.start() 
    # p1.join()
    # p2.join() 

if __name__=='__main__':
    get_data()
    # main()
        
