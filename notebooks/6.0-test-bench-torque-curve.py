"""
This script is used to determine the speed torque curve of the test setup in order to learn the maximum opperating torque as well as the optimal opperating conditions for gear crack growth

Notice that a different convention is used for the different channels measured
Channel 6: Oil temperature
Channel 7
"""

import Proc_Lib as proc
import pandas as pd
import matplotlib.pyplot as plt
import Minimum_Crack_Size as min_crack

plt.close("all")

filename = "Torque_cal_1_300Hz"
#filename = "Cycle_10_end"

dir = 'C:\\Users\\u15012639\\Desktop\\DATA\\h5_datasets' + "\\" + filename + ".h5"
df = pd.read_hdf(dir)
d = proc.Dataset(df, proc.Bonfiglioli)

print(d.dataset.keys())
print(d.derived_attributes.keys())

#d.plot_TSA()


d.plot_time_series("Torque")
plt.title("Motor current draw measured in Volt")

plt.figure()
d.plot_time_series("1PR_Mag_Pickup")
plt.title("Magnetic Pickup")

plt.figure()
d.plot_time_series("T_amb")
plt.title("Strain Gauge Torque Measurement")

plt.figure()
d.plot_time_series("T_oil")
plt.title("T oil")


# Operating condition parameters
##################
Pressure_angle = 20
Number_of_ring_gear_teeth = 62
Number_of_sun_teeth = 13
Number_of_planet_gear_teeth = 24
Module = 2.3e-3 #[m] Module of 2.3mm is very uncommon. Usually 2.5. However, this is what I measured
Motor_speed  = 1200 #RPM
opper = min_crack.Opperating_Conditions(Pressure_angle,
                              Number_of_ring_gear_teeth,
                              Number_of_sun_teeth,
                              Number_of_planet_gear_teeth,
                              Module,
                              Motor_speed)


opper.T = 4.1/proc.Bonfiglioli.GR # Divide the measured torque by the gear ratio to get the sun gear torque as used in the tooth force calculations

print(opper.Planet_tooth_force)
opper.Planet_tooth_force = opper.F_PS()

print("Planet tooth force: ", opper.Planet_tooth_force, "[N]")


#Current datapoint for reference
force = 5.5e3 #N
thickness = 4 #mm
cycles = 6*60*32

force_new = 45
thickness_new = 4


force_ratio = force_new/force
thickness_ratio = thickness_new/thickness

m=2.25
cycle_scale = (thickness_ratio/force_ratio)**m

print("cycle scaled with :",cycle_scale)

# @ 60Hz
cycles_required = cycles*cycle_scale
time = cycles_required/6

print("Time required [Hrs]: " ,time/3600)
print("Time required [years]: " ,time/(3600*24*365))



plt.figure()
plt.scatter(d.derived_attributes["rpm_mag"]*proc.Bonfiglioli.GR, d.dataset["T_amb"])
plt.ylabel("Torque [Nm]")
plt.xlabel("RPM")


print("Fault frequency @ 1200 RPM:", opper.F_planet_fault())