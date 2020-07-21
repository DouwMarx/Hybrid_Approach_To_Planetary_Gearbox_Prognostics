import src.data.make_data.data_proc as dproc

data_dir = "D:\\M_Data\\raw\\G2"
#data_dir = "D:\\M_Data\\raw\\G_Quick_Iter"

dproc.make_h5_for_dir_contents(data_dir, split=True)

