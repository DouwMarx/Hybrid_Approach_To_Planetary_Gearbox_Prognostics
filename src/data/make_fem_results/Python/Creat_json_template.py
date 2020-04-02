import json
import numpy as np


# Mesh
ring_mesh = "Ring_Marc_Mesh_0305"
#planet_bdf = "Planet_Marc_Mesh_halfcrack.bdf"
#planet_bdf = "Planet_Marc_Mesh_0311"
crack_length = 2.5

# Loadcase
total_rotation = 0.32 # The total angular distance rotated [rad]
n_increments = 10
applied_moment = 4 # Nm

# Contact
friction_coefficient = 0.07 # Dynamic friction coefficient for lubricated Cast iron on Cast iron https://www.engineeringtoolbox.com/friction-coefficients-d_778.html

# Geometry
gear_thickness = 12

move_planet_up = -1.65 # Distance the planet gear should be moved down [mm]
rotate_planet = -1.72 - (360 / 24) * 2 # Angle the planet gear should be rotated [degrees]
planet_carrier_pcr = 86.47 / 2 # Pitch Centre Radius of planet carrier axle

ring_gear_external_radius = 88.13  # External Radius of Ring gear [mm]
ring_gear_rotation = -(360 / 62) * 2  # Angle the ring gear should be rotated

planet_axle_radius = 29.32 / 2 #Internal radius of the planet gear [mm]

#  Material
E = 200  # MPa
v = 0.3



####Crack

# Loadcase
time = 11        # Set the number of steps to be the same as the time
n_steps = time
Applied_Load = 20

# Crack
crack_start_coord = [2.8, 68.5] # x,y coordinate of crack initiator start
crack_end_coord = [1.9, 68] # x,y coordinate of crack initiator start

fatigue_time_period = 2 #[seconds]
Maximum_Crack_Growth_Increment = 1 #[mm]
Paris_Law_Threshold = 0 #[MPa sqrt(mm)]
Paris_Law_C = 1e-09  # [m/(cycle*MPa m^0.5)]
Paris_Law_m = 2.25

Minimum_Growth_Increment = 0.5 #[mm]

# Geometry
gear_thickness = 12


R_carrier_axle_adjusted = 86.47 / 2 # Pitch Centre Radius of planet carrier axle

planet_axle_radius = 29.32 / 2 #Internal radius of the planet gear [mm]


#  Material
E = 200  # MPa
v = 0.3



#:load the exported json file
with open('dict.json') as json_file:
    dict = json.load(json_file)




with open("dict.json","w") as outfile:
    json.dump(dict, outfile, indent=8, separators=(',', ': '))



