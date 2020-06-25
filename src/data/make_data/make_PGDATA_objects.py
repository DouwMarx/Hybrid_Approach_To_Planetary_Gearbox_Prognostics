import os
import pandas as pd
from tqdm import tqdm
import pickle
import src.features.proc_lib as proc

import definitions

# directory = definitions.root + "\\data\\interim"
# directory = definitions.root + "\\data\\interim\\QuickIter"
# directory = definitions.root + "\\data\\interim\\G"
# directory = definitions.root + "\\data\\interim\\pre_wss_split"
# directory = definitions.root + "\\data\\interim\\pre_wss"

# directory = "D:\\M_Data\\interim\\pre_wss_full_crack_optim_speed"
# directory = "D:\\M_Data\\interim\\tooth_missing_single_planet"
directory = "D:\\M_Data\\interim\\g1_worst_case_crack"

for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder
    # try:
    df = pd.read_hdf(directory + "\\" + filename)
    first_meshing_tooth = 10
    d = proc.Dataset(df, proc.Bonfiglioli, filename, first_meshing_tooth)  # Create the dataset object

    with open(definitions.root + "\\data\\processed" + "\\" + filename[0:-3] + ".pgdata", 'wb') as config:
        pickle.dump(d, config)

# except:
#     print(filename, "gives problems - check tacho signal")
#     continue


# # Load the dataset object
#      with open("Cycle_10.PGDATA", 'rb') as config:
#         dd = pickle.load(config)
