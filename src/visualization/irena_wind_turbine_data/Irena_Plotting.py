# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:00:27 2019

@author: douwm
"""

import numpy as np 
import matplotlib.pyplot as plt

D = np.array([[16926,	 23964,	 30728,	 38667,	 47682,	 58405,	 73158,	 91542,	 115556,	 150122,	 180852,	 219981,	 266866,	 299994,	 349203,	 416211,	 466957,	 514747,	 563659],
              [30925,	 37716,	 52419,	 63007,	 83568,	 102677, 128906, 168103,	 215764,	 270953,	 342092,	 432480,	 524489,	 635110,	 712027,	 828251,	 954658,	1134451, np.NaN]])

Y = np.arange(2000,2019)

plt.figure()
#plt.plot(Y,D[0,:])
plt.bar(Y,D[0,:]/1000,width=0.8,color = "k")
plt.xticks(Y, Y, fontsize=7, rotation=90)
plt.ylabel("Global Wind Energy Electricity Capacity [GW]")
plt.xlabel("Year")
#tick_label = Y
plt.savefig("Images_1/Wind_Energy_Capacity.pdf")

Cap = D[0,:]
I = Cap[1+12:-1]/Cap[0+12:-2]
print(I)
A = np.average(I)

print("The average yearly increase in wind energy capacity over the past 18 years is {} %: ".format(np.round(100*(A-1),2)))
