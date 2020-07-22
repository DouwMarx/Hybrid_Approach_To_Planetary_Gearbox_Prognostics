import pickle
import matplotlib.pyplot as plt


# %% .pgdata is a serialized python object that contains both the dataset, some dataset attributes and methods that can
# be applied to the dataset %%

# %% load the dataset %%
filename = "g1_fc_1000.pgdata" # fc: full crack, 1000: rough indication of motor speed
path = filename
with open(path, 'rb') as filename:
    data = pickle.load(filename)

# %% use dir(data) to find all the attributes and methods of the dataset %%
print(dir(data))

# %% data.dataset is a pandas data frame with all channels measured %%
pd_dset = data.dataset
print(pd_dset.head())

# %% data.info and data.derived_attributes are dictionaries with pre-computed info %%
print("info: ", data.info.keys())  # "rpm_sun_ave" is average motor RPM
print("derived attributes", data.derived_attributes.keys())  # "trigger_time_mag" is timestamps planet passes transducer
                                                             # "PPF_ave" is average planet pass frequency (season)
                                                             # Rotation frequency of planet: data.PG.f_p(1/data.info["carrier_period_ave"]) (day)

# %% information and functions for the gearbox for instance number of teeth %%
print(dir(data.PG))

# %% dataset supports some plotting %%
# plt.figure()
# data.plot_time_series("Acc_Sun")
#
# plt.figure()
# data.plot_time_series("1PR_Mag_Pickup")
#
# data.plot_rpm_over_time()

