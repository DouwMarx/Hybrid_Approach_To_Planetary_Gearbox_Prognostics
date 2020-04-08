import pickle
import src.features.proc_lib as proc
import definitions

Zr = 62
Zs = 13
Zp = 24

Bonfiglioli = proc.PG(Zr,Zs,Zp)


#Load the dataset object
filename = "g1_p0_v10_0.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename

with open(directory, 'rb') as filename:
    data = pickle.load(filename)


data.plot_fft("Acc_Sun",plot_gmf = True)
