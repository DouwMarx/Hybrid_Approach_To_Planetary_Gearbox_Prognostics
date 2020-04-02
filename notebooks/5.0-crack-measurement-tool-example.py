import os
import numpy as np
import definitions

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
    script_path = definitions.root + "\\src\\data\\Crack_Measurement.py"
    os.system('python ' + script_path + ' --image ' + image_path + " --w " + str(slot_width) + " --l " + str(slot_length) +" > crack_length.txt")

    # Load the crack length into memory
    log = open(r"crack_length.txt","r")
    log = log.readline()[0:-2]
    length = np.float(log)

    os.system('del crack_length.txt')

    return length



image_name = definitions.root + "\\data\\raw\\crack_measurement_photos\\crack_test_im.jpg"

l = compute_crack_length(image_name, 0.3, 1.2)

print(l)