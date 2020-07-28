import os
from tqdm import tqdm
import src.features.proc_lib as proc
import src.data.make_data.data_proc as dproc
import definitions
import pickle
import numpy as np

# Directory with all split up .h5 files
h5_dir = "D:\\M_Data\\interim\\G2\\g2_p5"
file_name = "g2_p5.h5"

# Make sure the full dataset was converted to .pgdata allready so the tooth position info can be extracted
# dproc.make_pgdata("D:\\M_Data\\interim\\G2\\g2_full","g2_p5.h5",proc.Bonfiglioli,10)

directory = definitions.root + "\\data\\processed\\" + file_name[0:-3] + ".pgdata"
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

# voltage_tested = ["9.8", "9.6", "9.4", "9.2", "9.0", "8.8"]
voltage_tested = ["9.8", "9.6", "9.4", "9.2", "9.0", "8.8"]

# 30sec dead time before tests starts, then 2.5min per test
min_time = 30 / 60 + np.arange(0, len(voltage_tested) + 1) * (2 + 30 / 60)
sec_time = min_time * 60  # Convert the above statement to seconds

discard_at_end_of_interval = 1  # Amount of seconds of data to discard before the new operating condition was
# supposed to be set.
discard_at_beginning_of_interval = 11  # Amount of seconds of data to be discarded at the beginning of a new cycle
# These discarded sections should account for time to set new voltage and reaching steady state

dataset_start_time = sec_time[0:-1] + discard_at_beginning_of_interval
dataset_end_time = sec_time[1:] - discard_at_end_of_interval

for d_set, voltage in zip(range(len(voltage_tested)), voltage_tested):
    h5_file_name = "g2_p5_" + voltage + ".h5"

    start_time = dataset_start_time[d_set]
    planet_pass_times = data.derived_attributes["trigger_time_mag"]
    planet_mesh_seq = data.derived_attributes["mesh_sequence_at_planet_pass"]

    # Which planet pass event occurs first after the dataset starts?
    first_planet_pass_of_dset = np.argmax(planet_pass_times > start_time)
    first_tooth_mesh = planet_mesh_seq[first_planet_pass_of_dset]  # The first tooth number to mesh for the dataset

    dproc.make_pgdata(h5_dir, h5_file_name, proc.Bonfiglioli, first_tooth_mesh)
