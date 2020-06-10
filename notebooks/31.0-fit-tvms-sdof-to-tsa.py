import numpy as np
import matplotlib.pyplot as plt
import src.models.lumped_mas_model as pglmm
import pickle
import definitions
import src.models.diagnostics as diag
import src.features.proc_lib as proc
plt.close("all")

filename = "g1_p0_v8_2.pgdata"
# filename = "cycle_5_end.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename

def make_data():
    #  Load the dataset object
    with open(directory, 'rb') as filename:
        data = pickle.load(filename)

    #data.plot_rpm_over_time()
    tsa_obj = proc.Time_Synchronous_Averaging()

    tsa_obj.info = data.info
    tsa_obj.derived_attributes = data.derived_attributes
    tsa_obj.dataset = data.dataset
    tsa_obj.dataset_name = data.dataset_name
    tsa_obj.PG = data.PG

    offset_frac = (1/62)*(0.5)
    winds = tsa_obj.window_extract(offset_frac, 2*1/62, "Acc_Carrier", plot=False)
    wind_ave = tsa_obj.window_average(winds,plot=False)

    tooth = wind_ave[2,:] # 2 Gear mesh periods
    tooth = tooth[66:224] # A (carefully selected) single gear mesh period
    np.save(directory[0:],tooth)

# Load the save signal section for quick iteration
d = np.load(directory + ".npy")

# Plot the original tsa data
#plt.figure()
t_orig = np.linspace(0,len(d)/38600,len(d))
#t_range = np.linspace(t_orig[0],t_orig[-1],1000)
#plt.plot(t_orig, d, "-o")

# Default model parameters
m = 0.1938
k_mean = 0.1*10e5
c =(k_mean + m)*0.03
F = 100

delta_k =  0.5*k_mean

#X0 = np.array([0, 0])
X0 = np.array([-F / (k_mean + delta_k), 0])


t_step_start = 0
t_step_duration = 5/38600

sdof_dict = {"m": m,
             "c": c,
             "F": F,
             "delta_k": delta_k,
             "k_mean": k_mean,
             "X0": X0,
             "t_range": t_orig,
             "t_step_start": t_step_start,
             "t_step_duration": t_step_duration,
             "tvms_type": "sine_mean_delta_drop"}

#high_res_start_dict = sdof_dict
#high_res_start_dict["t_range"] = t_orig

#startpoint_sys = pglmm.SimpleHarmonicOscillator(high_res_start_dict) # initiate a lumped mass model
#t,startpoint_sys_sol = startpoint_sys.get_transducer_vibration() # Find the solution to the lmm
#plt.plot(t_range,startpoint_sys_sol)

optfor = {"c": [0.1*c, 10*c],
           "F": [0.1*F, 100*F],
            "delta_k": [0.01*delta_k , 100*delta_k],
           "k_mean": [0.01*k_mean, 100*k_mean]}

diag_obj = diag.Diagnostics(d,optfor,sdof_dict,pglmm.SimpleHarmonicOscillator)
diag_obj.plot_fit()

s = diag_obj.do_optimisation()
#print(s)
diag_obj.plot_fit()
