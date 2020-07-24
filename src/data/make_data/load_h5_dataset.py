"""
This file is used to make .pgdata objects from a series of .h5 files in a folder
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd

# define directory
directory = "D:\\M_Data\\interim\\QuickIter"
filename = "g1_fc_1000.h5"

# Load the dataset as a pandas dataframe
df = pd.read_hdf(directory + "\\" + filename)

# The channels of that dataset
print(df.keys())

# Accessing a channel as np array
acc = df["Acc_Carrier"].values

