import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions
import scipy.integrate as inter
import scipy.optimize as opt

import src.features.proc_lib as proc
plt.close("all")

#  Load the dataset object
filename = "g1_p0_v9_0.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)


tobj = proc.TransientAnalysis()
tobj.info = data.info
tobj.derived_attributes = data.derived_attributes
tobj.dataset = data.dataset
tobj.dataset_name = data.dataset_name

windows = data.derived_attributes["extracted_windows"]

n = np.random.randint(0, np.shape(windows)[0])
sig = windows[n, :]

trans,peak,tgm  = tobj.get_transients(sig)

def freq_int(x,fs,ints):
    "Performs frequency intergration"
    h = np.fft.fft(x)
    n = len(h)

    w = np.fft.fftfreq(n,d=1/fs)*2*np.pi
    w[0] = 1

    w = w*1j

    g = np.divide(h,w**ints)
    y = np.fft.ifft(g)
    y = np.real(y)
    return(y)

def integrate_signal(acc_sig,int_type = "time"):
    if int_type == "time":
        vel = inter.cumtrapz(acc_sig,None,dx=1/data.info["f_s"])
        disp = inter.cumtrapz(vel,None,1/data.info["f_s"])

        return vel,disp

    if int_type == "frequency":
        vel = freq_int(acc_sig,data.info["f_s"],1)
        disp = freq_int(acc_sig,data.info["f_s"],2)
        return vel,disp

fig,axs = plt.subplots(2,3)

i = np.random.randint(np.shape(trans)[0])
acc = trans[i, :]
for int_type,i in zip(["time","frequency"],[0,1]):

    axs[i,0].plot(acc, label = "acc")
    vel,disp = integrate_signal(acc, int_type=int_type)

    axs[i,1].plot(vel, label = "vel")

    axs[i,2].plot(disp, label = "disp")


#va,da = integrate_signal(trans)

#print(va.shape)

#plt.figure()
#plt.plot(va[i,:])

class one_dof_sys(object):
    """ Used to fit 1DOF spring mass damper system to Acceleration signal"""
    def __init__(self,acc_sig,fs):
        self.fs = fs
        self.acc = acc_sig
        self.vel, self.disp = self.integrate_signal(self.acc, self.fs,int_type="frequency")

        self.delta_k = 1e6

        self.acc = self.acc - np.mean(self.acc)
        self.vel = self.vel - np.mean(self.vel)
        self.disp = self.disp - np.mean(self.disp)

    def freq_int(self,x, fs, ints):
        "Performs frequency intergration"
        h = np.fft.fft(x)
        n = len(h)

        w = np.fft.fftfreq(n, d=1 / fs) * 2 * np.pi
        w[0] = 1

        w = w * 1j

        g = np.divide(h, w ** ints)
        y = np.fft.ifft(g)
        y = np.real(y)
        return (y)


    def integrate_signal(self,acc_sig,fs, int_type="time"):
        if int_type == "time":
            vel = inter.cumtrapz(acc_sig, None, dx=1 / fs)
            disp = inter.cumtrapz(vel, None, dx = 1 / fs)

            return vel, disp

        if int_type == "frequency":
            vel = freq_int(acc_sig, fs, 1)
            disp = freq_int(acc_sig, fs, 2)
            return vel, disp

    def cost(self,theta):
        """
        Cost funcion for determining model parameters
        Parameters
        ----------
        #theta = [m,c,k,F]
        theta = [m,c,F]

        Returns
        -------

        """
       # return np.linalg.norm(theta[0]*self.acc + theta[1]*self.disp + theta[2]*self.vel - theta[3])
        return np.linalg.norm(theta[0]*self.acc + self.delta_k*self.disp + theta[1]*self.vel - theta[2])

    def run_optimisation(self):
        #bnds = ((0, None), (0, None),(0,None),(0,None))
        #s = opt.minimize(self.cost,np.ones(4),bounds=bnds)

        bnds = np.array([[1,100],  #m
                         [1,100],  #c
                         #[1,100],  #k
                         [1,100]]) #F

        s = opt.differential_evolution(self.cost,bounds=bnds)
        return s

ods = one_dof_sys(acc,data.info['f_s'])
print(ods.run_optimisation())
