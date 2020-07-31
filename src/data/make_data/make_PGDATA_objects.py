"""
This file is used to make .pgdata objects from a series of .h5 files in a folder
"""
import os
from tqdm import tqdm
import src.features.proc_lib as proc
import src.data.make_data.data_proc as dproc

# Define a directory of .h5 files to convert to .pgdata
# ======================================================================================================================
# directory = "D:\\M_Data\\interim\\pre_wss_full_crack_optim_speed"
directory = "D:\\M_Data\\interim\\tooth_missing_single_planet"
# directory = "D:\\M_Data\\interim\\G1_full_crack_opt_speed"
# directory = "D:\\M_Data\\interim\\QuickIter"
# directory = "D:\\M_Data\\interim\\Find_Mag_Pickup_Trigger_Loc"

first_meshing_tooth = 10  # Define the gear tooth that the transducer would directly measure when the planet gear
# passes it the first time.

for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder
    dproc.make_pgdata(directory, filename, proc.Bonfiglioli, first_meshing_tooth)
