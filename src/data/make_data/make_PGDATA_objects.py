import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import importlib
import Proc_Lib
importlib.reload(Proc_Lib)
import pickle
import time

import Proc_Lib as proc

directory = 'D:\h5_datasets'

for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder

    dir = 'D:\h5_datasets' + "\\" + filename #+ ".h5"
    df = pd.read_hdf(dir)

    d = proc.Dataset(df, proc.Bonfiglioli)

    with open('D:\pickle_datasets' + "\\" +filename[0:-3] + ".PGDATA", 'wb') as config:
        pickle.dump(d, config)

    continue





# # Load the dataset object
  #      with open("Cycle_10.PGDATA", 'rb') as config:
   #         dd = pickle.load(config)