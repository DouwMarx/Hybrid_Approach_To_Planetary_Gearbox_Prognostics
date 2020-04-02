import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm


def main():

    #directory = 'D:\Datasets'
    directory = "C:\\Users\\u15012639\\Desktop\\DATA"

    for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder
        if filename.endswith('.xlsx'):   # This section deals with .xlsx files
            dir = directory + '\\' + filename
            df = pd.read_excel(dir, header = 1)  # Set the headers of the dataframe to be line 2 in excel sheet
            df = df.drop(range(47), axis=0)   # Remove blank lines and units of channels
            df.columns = ['Time', 'Acc_Carrier', 'Acc_Sun', 'Tacho_Carrier', 'Tacho_Sun', '1PR_Mag_Pickup','T_amb', 'T_oil', 'Torque']
            df = df.reset_index(drop=True)  # Makes sure that the indexes start from zero
            df = df.astype("float")  # Changes the data type to float

            #save_dir = 'D:\h5_datasets' + "\\" + filename[0:-5].lower() + ".h5"
            save_dir = 'C:\\Users\\u15012639\\Desktop\\DATA\\h5_datasets' + "\\" + filename[0:-5].lower() + ".h5"
            df.to_hdf(save_dir, key="df" , mode="w")
            continue

        if filename.endswith('.MAT'):   # This section deals with .MAT files
            dir = directory + '\\' + filename
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

            save_dir = 'C:\\Users\\u15012639\\Desktop\\DATA\\h5_datasets' + "\\" + filename[0:-4].lower() + ".h5"
            df.to_hdf(save_dir, key="df" , mode="w")
            continue

        else:
            continue

if __name__ == "__main__":
    main()
