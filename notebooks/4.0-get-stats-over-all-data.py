import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os


collection = []

import definitions

directory = definitions.root + "\\data\\processed"

for filename in tqdm(os.listdir(directory)):  # Loop through all of the files in a folder
    if filename.startswith('g'):   # This section deals with .xlsx files
        with open(directory + "\\" + filename, 'rb') as name:
            dd = pickle.load(name)

            parameter_of_interest = dd.info["duration"]
            collection.append(parameter_of_interest)

plt.figure()
plt.hist(collection)