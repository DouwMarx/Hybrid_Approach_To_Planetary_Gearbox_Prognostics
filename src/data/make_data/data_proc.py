import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
from pathlib import Path

import definitions



#    directory = "D:\\M_Data\\raw\\pre_wss_G1_full_crack_opt_speed"

#    save_dir = "D:\\M_Data\\interim\\pre_wss_full_crack_optim_speed"

#    for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder

def make_h5(data_dir, filename,save_dir):
    if filename.endswith('.xlsx'):   # This section deals with .xlsx files
        dir = data_dir + '\\' + filename
        df = pd.read_excel(dir, header = 1)  # Set the headers of the dataframe to be line 2 in excel sheet
        df = df.drop(range(47), axis=0)   # Remove blank lines and units of channels
        df.columns = ['Time', 'Acc_Carrier', 'Acc_Sun', 'Tacho_Carrier', 'Tacho_Sun', '1PR_Mag_Pickup','T_amb', 'T_oil', 'Torque']
        df = df.reset_index(drop=True)  # Makes sure that the indexes start from zero
        df = df.astype("float")  # Changes the data type to float


        save_dir = definitions.root + "\\data\\interim" + "\\" + filename[0:-5].lower() + ".h5"
        df.to_hdf(save_dir, key="df" , mode="w")

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
        # Make a for full dataset
        full_dir = interim_dir + "\\" + data_dir[13:].lower() + "_full"
        Path(full_dir).mkdir(parents=True, exist_ok=True)

        for filename in tqdm(os.listdir(data_dir)):  # Loop through all of the files in a folder
            if filename.endswith(".MAT"):
                # Make a folder for the datasets to be split
                Path(interim_dir + "\\" + filename[0:-12].lower()).mkdir(parents=True, exist_ok=True)
            # Create the full dataset h5 files
            make_h5(data_dir, filename, full_dir)
    return