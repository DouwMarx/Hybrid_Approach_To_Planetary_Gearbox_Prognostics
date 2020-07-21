import os
from tqdm import tqdm
import src.data.make_data.data_proc as dproc
directory = "D:\\M_Data\\interim\\G_Quick_Iter\\g_quick_iter_full"


# filename = "g1_p0.h5"
for filename in os.listdir(directory):  # Loop through all of the files in a folder
    dproc.split_h5(filename,directory)

