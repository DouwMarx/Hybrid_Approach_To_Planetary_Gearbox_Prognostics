# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:17:50 2020

@author: u15012639
"""

import numpy as np

class Utilities(object):
    
    @classmethod
    def circle_boundary(self,centre,radius,location):
        """Checks if a node falls inside or outside a circle with given centre and radius
        
        centre : list with [x_centre,y_centre]
        radius : radius in mm
        """
        d = self.distance(centre,location)
        
        if d<radius:
            return True
        
        else:
            return False
        
        
        return d
    
    def distance(coord_1,coord_2):
        """Computes the distance between two points"""
        
        return np.sqrt((coord_1[0] - coord_2[0])**2 + (coord_1[1] - coord_2[1])**2)
    
#print(Utilities.distance([1,0],[2,0]))


print(Utilities.circle_boundary([0,0],1,[0.1,0]))