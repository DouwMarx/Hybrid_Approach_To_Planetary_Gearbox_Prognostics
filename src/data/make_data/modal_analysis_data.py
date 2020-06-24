import numpy as np
import matplotlib.pyplot as plt

# 432  : 1000 Hz hard tip
# 433  : 1000 Hz soft tip
# 434  : 11000 Hz hard tip

# Accelerometer? SN179969

for channel in ["ch2","ch3","ch4"]:
    directory = "D:\\M_Data\\raw\\modal_analysis\\" + "SIG0434_H(" + channel + ",ch1).csv"
    d = np.genfromtxt(directory, skip_header=31, delimiter=",")

    freq = d[:, 0]
    real = d[:, 1]
    imag = d[:, 2]

    mag = np.sqrt(real ** 2 + imag ** 2)

    plt.figure()
    plt.plot(freq, mag)
