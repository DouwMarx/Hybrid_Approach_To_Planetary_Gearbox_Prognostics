import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
from pathlib import Path
import src.features.proc_lib as proc
import pickle
import definitions

def make_h5(data_dir, filename,save_dir):
    if filename.endswith('.xlsx'):   # This section deals with .xlsx files
        dir = data_dir + '\\' + filename
        df = pd.read_excel(dir, header = 1)  # Set the headers of the dataframe to be line 2 in excel sheet
        df = df.drop(range(47), axis=0)   # Remove blank lines and units of channels
        df.columns = ['Time', 'Acc_Carrier', 'Acc_Sun', 'Tacho_Carrier', 'Tacho_Sun', '1PR_Mag_Pickup','T_amb', 'T_oil', 'Torque']
        df = df.reset_index(drop=True)  # Makes sure that the indexes start from zero
        df = df.astype("float")  # Changes the data type to float


        # save_dir = definitions.root + "\\data\\interim" + "\\" + filename[0:-5].lower() + ".h5"
        # df.to_hdf(save_dir, key="df" , mode="w")

        df.to_hdf(save_dir + "\\" + filename[0:-12].lower() + ".h5", key="df", mode="w")

    if filename.endswith('.MAT'):   # This section deals with .MAT files
        dir = data_dir + '\\' + filename
        mat = loadmat(dir)  # load mat-file
        mat.keys()

        df = pd.DataFrame(data=np.hstack((mat['Channel_1_Data'],
                                          mat['Channel_2_Data'],
                                          mat['Channel_3_Data'],
                                          mat['Channel_4_Data'],
                                          mat['Channel_5_Data'],
                                          mat['Channel_6_Data'],
                                          mat['Channel_7_Data'],
                                          mat['Channel_8_Data'],
                                          mat['Channel_9_Data'])))
        df.columns = ['Time', 'Acc_Carrier', 'Acc_Sun', 'Tacho_Carrier', 'Tacho_Sun', '1PR_Mag_Pickup', 'T_amb',
                      'T_oil', 'Torque']

        df.to_hdf(save_dir+ "\\" + filename[0:-12].lower() + ".h5", key="df" , mode="w")
    return

def make_h5_for_dir_contents(data_dir,split = False):
    """
    Makes h5 dataset but also creates a folder for the interim data if required
    Parameters
    ----------
    data_dir

    split: Boolean
           whether or not to split the dataset into separate datasets

    Returns
    -------

    """
    interim_dir = data_dir[0:10] + "interim" + data_dir[13:]
    # Create similarly named interim directory if it does not exist
    Path(interim_dir).mkdir(parents=True, exist_ok=True)

    if split == False:
        for filename in tqdm(os.listdir(data_dir)):  # Loop through all of the files in a folder
            make_h5(data_dir, filename, interim_dir)

    if split ==True:
        # Make a directory for full dataset
        full_dir = interim_dir + "\\" + data_dir[13:].lower() + "_full"
        Path(full_dir).mkdir(parents=True, exist_ok=True)

        print("Start converting full datasets to .h5")
        for filename in tqdm(os.listdir(data_dir)): # Make each of the full datasets and save them
            if filename.endswith(".MAT"):
                # Make a folder for the datasets to be split
                Path(interim_dir + "\\" + filename[0:-12].lower()).mkdir(parents=True, exist_ok=True)
            # Create the full dataset h5 files
            make_h5(data_dir, filename, full_dir)

        print("Start splitting full datasets")
        for filename in tqdm(os.listdir(full_dir)):  # Loop through all of the files in a folder
            split_h5(filename, full_dir)
        print("Splitting progress:")
    return

def split_h5(filename, directory):
    df = pd.read_hdf(directory + "\\" + filename)
    gear = "G1"
    print(" ")
    print("start splitting file: " + filename)

    fs = 38400  # Sampling rate

    voltage_tested = ["9.8", "9.6", "9.4", "9.2", "9.0", "8.8"]

    # 30sec dead time before tests starts, then 2.5min per test
    min_time = 30 / 60 + np.arange(0, len(voltage_tested) + 1) * (2 + 30 / 60)
    sec_time = min_time * 60  # Convert the above statement to seconds

    discard_at_end_of_interval = 1  # Amount of seconds of data to discard before the new operating condition was supposed to be set.
    discard_at_beginning_of_interval = 12  # Amount of seconds of data to be discarded at the beginning of a new cycle
    # Should account for time to set new voltage and reaching steady state

    dataset_start_time = sec_time[0:-1] + discard_at_beginning_of_interval
    dataset_end_time = sec_time[1:] - discard_at_end_of_interval

    for i in tqdm(range(len(voltage_tested))):  # Loop trough respective torque settings
                                          start_sample = int(dataset_start_time[i] * fs)  # Find the appropriate index by multiplying by fs
                                          end_sample = int(dataset_end_time[i] * fs)
                                          df_chunk = df.loc[start_sample:end_sample, :]

                                          p = os.path.dirname(directory)
                                          rest_of_path =  filename[0:-3].lower() + "\\" + filename[0:-3].lower() + "_" + str(voltage_tested[i]) + ".h5"
                                          save_dir = os.path.join(p,rest_of_path)
                                          df_chunk.to_hdf(save_dir, key="df", mode="w")
    return

def make_pgdata(h5_dir, h5_name, gearbox_geometry, first_meshing_tooth):
    df = pd.read_hdf(h5_dir + "\\" + h5_name)
    d = proc.Dataset(df, gearbox_geometry, h5_name, first_meshing_tooth)  # Create the dataset object

    with open(definitions.root + "\\data\\processed" + "\\" + h5_name[0:-3] + ".pgdata", 'wb') as config:
        pickle.dump(d, config)
    return

