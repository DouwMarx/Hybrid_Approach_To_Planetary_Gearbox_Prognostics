
import matplotlib.pyplot as plt
import pandas as pd
import src.features.proc_lib as proc
import definitions
import pickle

#  Load the dataset object
filename = "motor_fan.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    motor = pickle.load(filename)

    plt.figure()
    motor.plot_fft("Acc_Sun")
    plt.title("Motor Fan Only Frequency Response")


#  Load the dataset object
filename = "pump_fan.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    pump = pickle.load(filename)

    plt.figure()
    pump.plot_fft("Acc_Sun")
    plt.title("Pump Fan Only Frequency Response")
