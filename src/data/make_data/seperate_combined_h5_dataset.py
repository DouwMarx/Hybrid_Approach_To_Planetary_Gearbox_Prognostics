import os
import pandas as pd
from tqdm import tqdm
import pickle
import src.features.proc_lib as proc
import definitions
import numpy as np

#directory = definitions.root + "\\data\\interim\\QuickIter"
directory = definitions.root + "\\data\\interim\\pre_wss"

#filename = "g1_p0.h5"
for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder
    df = pd.read_hdf(directory + "\\" + filename)

    #        with open(definitions.root + "\\data\\processed" + "\\" + filename[0:-3] + ".pgdata", 'wb') as config:
    #            pickle.dump(d, config)

    fs = 38400
    min_time = 30/60 + np.arange(0,7)*(2+30/60)  # 30sec dead time before tests starts, then 2.5min per test
    sec_time = min_time*60 # Convert the above statement to seconds

    discard_at_end_of_interval = 1 # Amount of seconds of data to discard before the new operating condition was supposed to be set.
    discard_at_beginning_of_interval = 12 # Amount of seconds of data to be discarded at the beginning of a new cycle
                                          # Should account for time to set new voltage and reaching steady state

    dataset_start_time = sec_time[0:-1] + discard_at_beginning_of_interval
    dataset_end_time = sec_time[1:] - discard_at_end_of_interval

    voltage_names = ["9.8","9.6","9.4","9.2","9.0","8.8"]

    for i in range(6): # Loop trough respective torque settings
        start_sample = int(dataset_start_time[i]*fs)
        end_sample = int(dataset_end_time[i]*fs)
        df_chunck = df.loc[start_sample:end_sample,:]

        save_dir = definitions.root + "\\data\\interim" + "\\pre_wss_split\\" + filename[0:-3].lower() + "_" + voltage_names[i] + ".h5"
        df_chunck.to_hdf(save_dir, key="df" , mode="w")
