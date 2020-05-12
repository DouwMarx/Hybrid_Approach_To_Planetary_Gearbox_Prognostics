
import models.lumped_mas_model.llm_models as lmm
import numpy as np
import matplotlib.pyplot as plt
import definitions
import src.models.lumped_mas_model as pglmm
import dill

plt.close("all")



def save_model():
    with open(definitions.root + "\\models\\solved_models" + "\\" + "get_y1" + ".lmmsol", 'wb') as config:
        PG = lmm.make_chaari_2006_model()
        PG.get_solution()
        dill.dump(PG, config)
    return


def load_model():
        with open(definitions.root + "\\models\\solved_models" + "\\" + "get_y1" + ".lmmsol", 'rb') as config:
            PG = dill.load(config)
        return PG

def window_extract(wind_length, y, Omega_c, fs):
    """
    Extracts windows of length l samples every 2*pi/udot_c seconds
    In  other words this extracts a window of samples as a planet gear passes the transducer
     """

    if wind_length % 2 is not 0 == True:
        raise ValueError("Please enter uneven window length")

    samples_per_rev = int((1/2*np.pi)*(1/Omega_c)*fs)
    print("fs ", fs)
    print("samples per rev ", samples_per_rev)
    window_center_index = np.arange(0, len(y), samples_per_rev).astype(int)

    n_revs = np.shape(window_center_index)[0] - 2  # exclude the first and last revolution to prevent errors with insufficient window length
                                                   # first window would have given problems requireing negative sample indexes

    windows = np.zeros((n_revs, wind_length))  # Initialize an empty array that will hold the extracted windows
    window_half_length = int(np.floor(wind_length/2))

    window_count = 0
    for index in window_center_index[1:-1]:  # exclude the first and last revolution to prevent errors with insufficient window length
        windows[window_count, :] = y[index - window_half_length:index + window_half_length + 1]
        window_count += 1
    return windows

save_model()

PG = load_model()

winds = PG.get_windows(51)
#PG.plot_solution("Displacement")

# transp = pglmm.Transmission_Path(PG)
# y = transp.y()
# plt.figure()
# plt.plot(y)
#
# fs = 1/np.average(np.diff(PG.time_range))
#
# winds = window_extract(5001, y, 2*np.pi*1000/60, fs)
#
#
plt.figure()
plt.plot(winds[0, :])