import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import src.data.make_data.data_proc as dproc
import definitions

data_dir = "D:\\M_Data\\raw\\G2"
# save_dir = "D:\\M_Data\\interim\\pre_wss_full_crack_optim_speed"
#
# for filename in tqdm(os.listdir(data_dir)):  # Loop through all of the files in a folder
#     dproc.make_h5(data_dir, filename, save_dir)

dproc.make_h5_for_dir_contents(data_dir, split=True)

