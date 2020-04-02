import os
import numpy as np
import pandas as pd
import Proc_Lib as proc




def compute_crack_length(image_path,slot_width, slot_length):
    """
    Run crack measurement sofware from Crack_Measurement.py and return the crack length
    Parameters
    ----------
    image_path
    slot_width
    slot_length

    Returns
    -------

    """


    #Run the crack measurement software and write the resulting crack length to a file
    os.system('python Crack_Measurement.py --image ' + image_path +  " --w " + str(slot_width) + " --l " + str(slot_length) +" > crack_length.txt")

    # Load the crack length into memory
    log = open(r"crack_length.txt","r")
    log = log.readline()[0:-2]
    length = np.float(log)

    os.system('del crack_length.txt')

    return length


#image_name = proc.Z_crack_im_path + "\\" + "Z_cycle_0"  #test_im.jpg"
image_name = "Z_Cycle_4d.jpg"

l = compute_crack_length(image_name, 0.3, 1.2)

print(l)