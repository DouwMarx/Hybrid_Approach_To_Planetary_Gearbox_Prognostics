import os
from tqdm import tqdm
import src.features.proc_lib as proc
import src.data.make_data.data_proc as dproc

# directory = "D:\\M_Data\\interim\\pre_wss_full_crack_optim_speed"
# directory = "D:\\M_Data\\interim\\tooth_missing_single_planet"
directory = "D:\\M_Data\\interim\\g1_worst_case_crack"

for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder
    dproc.make_pgdata(directory, filename, proc.Bonfiglioli, 10)
