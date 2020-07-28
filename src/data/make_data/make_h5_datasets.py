import src.data.make_data.data_proc as dproc

# data_dir = "D:\\M_Data\\raw\\G1_full_crack_opt_speed"
#data_dir = "D:\\M_Data\\raw\\G_Quick_Iter"
#data_dir = "D:\\M_Data\\raw\\Find_Mag_Pickup_Trigger_Loc"
data_dir = "D:\\M_Data\\raw\\G2"

dproc.make_h5_for_dir_contents(data_dir, split=True)
#dproc.make_h5_for_dir_contents(data_dir, split=False)
