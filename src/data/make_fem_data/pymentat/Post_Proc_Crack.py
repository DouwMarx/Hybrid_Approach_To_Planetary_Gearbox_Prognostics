# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:27:28 2020

@author: u15012639
"""

from py_mentat import *
import math

name = 'crack_script_dev_job1'
n_increm = 11

def open_file(name):
    py_send("*post_open ..\\Run_Dir\\" + name + ".t16")
    py_send("*post_next")
    py_send("*fill_view")
    py_send("*zoom_box")
    py_send("*zoom_box(2,0.472408,0.085288,0.525084,0.149254)")
    return

def extract_meshes():
    mesh_extract = True
    for i in range(n_increm):
        py_send("*post_next")
        if mesh_extract == True:
            py_send("*export nastran '..\Mesh\m1_a" + str(int(i/2)) + "mm.bdf' yes")

        mesh_extract = not mesh_extract

    return



def main():
    #py_send("*post_close")

    open_file(name)

    extract_meshes()




    return

if __name__ == "__main__":
    py_connect('',40007)
    main()
    py_disconnect()