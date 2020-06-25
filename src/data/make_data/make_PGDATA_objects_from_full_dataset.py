import os
import pandas as pd
from tqdm import tqdm
import pickle
import src.features.proc_lib as proc

import definitions

filename = 0
full_dataset_path = "D:\\M_Data\\interim\\g1_worst_case_crack"

df = pd.read_hdf(full_dataset_path)
first_meshing_tooth_full_dataset = 10
d = proc.Dataset(df, proc.Bonfiglioli, filename, first_meshing_tooth_full_dataset)  # Create the dataset object

with open(definitions.root + "\\data\\processed" + "\\" + filename[0:-3] + ".pgdata", 'wb') as config:
    pickle.dump(d, config)

# except:
#     print(filename, "gives problems - check tacho signal")
#     continue


# # Load the dataset object
#      with open("Cycle_10.PGDATA", 'rb') as config:
#         dd = pickle.load(config)
