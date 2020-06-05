
import os
import pandas as pd
from tqdm import tqdm
import pickle
import src.features.proc_lib as proc

import definitions

#directory = definitions.root + "\\data\\interim"
#directory = definitions.root + "\\data\\interim\\QuickIter"
directory = definitions.root + "\\data\\interim\\G"

for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder
    try:
        df = pd.read_hdf(directory + "\\" + filename)
        d = proc.Dataset(df, proc.Bonfiglioli,filename)  # Create the dataset object

        with open(definitions.root + "\\data\\processed" + "\\" + filename[0:-3] + ".pgdata", 'wb') as config:
            pickle.dump(d, config)

    except:
        print(filename, "gives problems - check tacho signal")
        continue



# # Load the dataset object
  #      with open("Cycle_10.PGDATA", 'rb') as config:
   #         dd = pickle.load(config)