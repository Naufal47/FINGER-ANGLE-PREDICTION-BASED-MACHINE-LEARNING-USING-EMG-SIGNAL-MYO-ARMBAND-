import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import myo
from threading import Lock, Thread
from libRMS import *
import joblib
# --------------------------------------------------------------------------------------
# PROGRAM PENGAMBIAN DATA SUDUT
# --------------------------------------------------------------------------------------
# ser = serial.Serial('com4',baudrate=2000000)
# data = []
# start_time = time.time()
# data_final=[]
# seconds = 55
collectt=True
model1 = joblib.load('SVM_naufal.pkl')
# realtime1 = []

# def flex():
    
#     global collectt
#     start_time = time.time()
#     while True:
#         current_time = time.time()
#         elapsed_time = current_time-start_time
        
#         arduino_data = ser.readline().decode('ascii', errors = 'ignore')
#         arduino_data = arduino_data.strip().split(" ")
#         # data.append(arduino_data)
        
#         if elapsed_time >= 1:
#             break
#     time.sleep(3)
#     start_time = time.time()
#     print('=========Mulai==========')
#     while collectt:
#         current_time = time.time()
#         elapsed_time = current_time-start_time
        
#         arduino_data = ser.readline().decode('ascii', errors = 'ignore')
#         arduino_data = arduino_data.strip().split(" ")
#         data.append(arduino_data)
#         t = int(elapsed_time)
#         print(t, end='\r')
#         if t==0 or t==15 or t==30 or t==45 or t==60 or t==75:
#             print('      =======> GENGGAM <========', end='\r' )
#         elif t==5 or t==20 or t==35 or t==50 or t==65 or t==80:
#             print('      =========>LEPASKAN<=======', end='\r')
#         elif t==10 or t==25 or t==40 or t==55 or t==70 or t==85:
#             print('      =========>ISTIRAHAT<======', end='\r')

#         if elapsed_time >= seconds:
#             collectt=False
#             break
#     data2=pd.DataFrame(data)
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#     data2.to_csv('true_flexSVM2.csv', index=False) #<------------NAMA FILE FLEX----------
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#     # print(len(data2))
#     print("========While Loop Berhenti=========")

# -----------------------------------------------------------------------------------
# PROGRAM PENGAMBILAN DATA EMG (MYO ARMBAND)
# -----------------------------------------------------------------------------------

class Plot(object):
    def __init__(self, listener):
        self.n = listener.n
        self.listener = listener
        # self.fig = plt.figure()
        # self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
        # [(ax.set_ylim([-100, 100])) for ax in self.axes]
        # self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
        # plt.ion()

    def main(self):
        pred = []
        while collectt:
            
            plt.pause(1/5) 
            emg_data = self.listener.get_emg_data()
            emg_data = np.array([x[1] for x in emg_data])
            emgX = pd.DataFrame(emg_data)

            oi = emgX.shape
            # print(oi)
            
            if oi[0] > 39:
                
                data = np.array(emgX).reshape(-1,8)
                save_emg = pd.DataFrame(data)
                save_emg.columns = ['ch1', 'ch2', 'ch3','ch4','ch5','ch6','ch7','ch8']
                datafitur = prep(save_emg)
                realtime = model1.predict(datafitur)
               
                print(realtime)
        #         pred.append(realtime)
               
        # pred1 = pd.DataFrame(np.array(pred).reshape(-1,5))
        # pred1.to_csv('predict_flexSVM2.csv')      
            
          
            # hasil = pd.DataFrame(pred)
            # hasil.to_csv('predict_flex.csv')
                # print(datafitur.shape)
            
            # for g, data in zip(self.graphs, emg_data):
            #     if len(data) < self.n:
            #         # Fill the left side with zeroes.
            #         data = np.concatenate([np.zeros(self.n - len(data)), data])
            #     g.set_ydata(data)
            # #plt.tight_layout()
            # plt.draw()
        

    # def main(self):
    #     while collectt:
    #         self.update_plot()
    #         plt.pause(1.0 / 30)

def main():
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(40)
    with hub.run_in_background(listener.on_event):
        Plot(listener).main()

# if __name__ == '__main__':
#     main()
# -------------------------------------------------------------------------------------
# RUN PROGRAM PENGAMBILAN DATA EMG DAN KEKUTAN UNTUK REGRESI
# -------------------------------------------------------------------------------------

# def get_data():
#     p1 =Thread(target=flex)
#     p2 =Thread(target=main)

#     p1.start()
#     p2.start() 
#     # p1.join()
#     # p2.join() 

if __name__=='__main__':
    # get_data()
    main()
        
