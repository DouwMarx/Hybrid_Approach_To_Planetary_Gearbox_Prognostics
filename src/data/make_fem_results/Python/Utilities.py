# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:17:50 2020

@author: u15012639
"""

#import numpy as np
from py_mentat import *

def circle_boundary(centre,radius,location):
    """Checks if a node falls inside (True) or outside (False) a circle with given centre and radius
    
    centre : list with [x_centre,y_centre]
    radius : radius in mm
    """
    d = distance(centre,location)
    
    if d < radius:
        return True
    
    else:
        return False
    return d



def distance(coord_1,coord_2):
    """Computes the distance between two points"""
    
    return ((coord_1[0] - coord_2[0])**2 + (coord_1[1] - coord_2[1])**2)**0.5

def find_axle_nodes():
    """Finds the node nearest to the specified location"""
    py_send("*renumber_all") # Renumber all of the nodes in the mesh
    number_of_nodes = py_get_int('nnodes()') # The number of nodes in the mesh
    print("Number of Nodes: ",number_of_nodes)
    for i in range(1,number_of_nodes+1):
        string_x = "node_x(%d)" %i
        x_coord = py_get_float(string_x) # Get the x_coordinate of a particular node
        
        string_y = "node_y(%d)" %i
        y_coord = py_get_float(string_y) # Get the x_coordinate of a particular node
        
        # Ring gear radius = 0.08813
        print(x_coord)
        print(y_coord)

    return


print(circle_boundary([0,0],1,[0.1,0]))