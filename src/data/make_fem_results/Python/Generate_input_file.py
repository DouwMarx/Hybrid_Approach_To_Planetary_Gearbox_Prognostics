# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:19:00 2020

@author: u15012639
"""

import json

def save_obj(obj, name ):
    with io.open('obj/' + name + '.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(obj,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str(str_))
    return

def load_obj(name ):
    with open('obj/' + name + '.json') as f:
        return json.load(f)
    
move_planet_up = -1.7 # Distance the planet gear should be moved down [mm]
rotate_planet = -1.7  # Angle the planet gear should be rotated [degrees]
planet_carrier_pcr = 86.47 / 2 + move_planet_up # Pitch Centre Radius of planet carrier axle
ring_gear_external_radius = 88.13  # External Radius of Ring gear [mm]
planet_axle_radius = 29.32 / 2 #Internal radius of the planet gear [mm]

# loadcase
total_rotation = 0.1 # The total angular distance rotated [rad]
Rotation_table = "motion_3" # The rotation table to use for the analysis

dictionary = {move_planet_up : -1.7  # Distance the planet gear should be moved down [mm]
              rotate_planet : -1.7  # Angle the planet gear should be rotated [degrees]
              planet_carrier_pcr : 86.47 / 2 + move_planet_up  # Pitch Centre Radius of planet carrier axle
              ring_gear_external_radius = 88.13  # External Radius of Ring gear [mm]
planet_axle_radius = 29.32 / 2 #Internal radius of the planet gear [mm]

# loadcase
total_rotation = 0.1 # The total angular distance rotated [rad]
Rotation_table = "motion_3" # The rotation table to use for the analysis}