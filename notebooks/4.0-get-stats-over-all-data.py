import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import Proc_Lib
import Proc_Lib as proc
from tqdm import tqdm
import pickle



importlib.reload(Proc_Lib)


sd_list = []
speedrangelist = []


total_cycle_count = 0
total_time = 0
for cycle in tqdm([1,2,3,4,5,6,7,8,9,10,11,12,13,14]):
#for cycle in tqdm([1,2,3,4,5]):

    filename = "cycle_" + str(cycle)
    #filename = "Z_cycle_" + str(cycle)
    dir = 'D:\pickle_datasets' + "\\" + filename + ".PGDATA"

    with open(dir, 'rb') as config:
        dd = pickle.load(config)

    #print([cycle, dd.info["n_fatigue_cycles"]])

    total_cycle_count += dd.info["n_fatigue_cycles"]
    total_time+= dd.info["duration"]

    # fs = proc.sampling_rate(df["Time"].values) #  Compute the sampling rate for the dataset
    #
    # mag_pickup = df["1PR_Mag_Pickup"].values
    # print(np.shape(mag_pickup))
    #
    # rpm, t = proc.getrpm(mag_pickup,fs,2,1,1,fs)  # Compute the RPM for the input shaft based on the shaft encoder
    # plt.figure()
    # plt.plot(t,rpm)
    # ave_rpm = np.mean(rpm)  # The average input (low speed side) over the tested interval
    # Sd_rpm = np.std(rpm)
    # print(Sd_rpm)
    # sd_list.append(Sd_rpm)
    # speedrangelist.append(np.max(rpm)-np.min(rpm))
print("total_fatigue_cycle_count: ",total_cycle_count )
print("Duration (hrs): ",total_time/(3600))

