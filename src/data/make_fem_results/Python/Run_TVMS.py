from datetime import time

from py_mentat import *
import time
import math
import os
import json


# Preliminary Calculations
########################################################################################################################
#name
def load_run_file():
    """Loads the .json file that defines the simulation parameters an is later used to store the important simulation results"""
    global run_file,\
        total_rotation,\
        ring_mesh, planet_meshes,\
        n_increments,\
        planet_bdf_dir,\
        applied_moment,\
        friction_coefficient,\
        gear_thickness,\
        move_planet_up ,\
        rotate_planet,\
        planet_carrier_pcr,\
        ring_gear_external_radius,\
        ring_gear_rotation,\
        planet_axle_radius,\
        E,\
        v

    run_file = py_get_string("run_file")
    run_file_path = "..\\Run_Files\\" + run_file

    # Open the input file
    with open(run_file_path) as json_file:
        input = json.load(json_file)["TVMS Properties"]

        # Simulation Parameters
        ########################################################################################################################
        # Mesh
        ring_mesh = input["Mesh"]["Ring Mesh"]
        planet_bdf_dir = input["Mesh"]["Planet Meshes"]

        # Loadcase
        total_rotation = input["Load Case"]["Total Rotation"] #The total angular distance rotated [rad]
        n_increments = input["Load Case"]["Number Of Loadsteps"] #int
        applied_moment = input["Load Case"]["Number Of Loadsteps"]  # Moment on planet gear [Nm]

        # Contact
        friction_coefficient = input["Contact"]["Friction Coefficient"]  # Dynamic friction coefficient for lubricated Cast iron on Cast iron https://www.engineeringtoolbox.com/friction-coefficients-d_778.html

        # Geometry
        gear_thickness =input["Geometry"]["Gear Thickness"] # [mm]

        move_planet_up = input["Geometry"]["Move Planet Up"]#-1.65  # Distance the planet gear should be moved up [mm]
        rotate_planet = input["Geometry"]["Rotate Planet"]#-1.72 - (360 / 24) * 2  # Angle the planet gear should be rotated [degrees]
        planet_carrier_pcr = input["Geometry"]["Planet Carrier Pitch Centre Radius"]  # Pitch Centre Radius of planet carrier axle

        ring_gear_external_radius = input["Geometry"]["Ring Gear External Radius"]  # External Radius of Ring gear [mm]
        ring_gear_rotation = input["Geometry"]["Ring Gear Rotation"] #-(360 / 62) * 2  # Angle the ring gear should be rotated

        planet_axle_radius = input["Geometry"]["Planet Axle Radius"]  # Internal radius of the planet gear [mm]

        #  Material
        E = input["Material"]["Young's Modulus"] # MPa
        v = input["Material"]["Poisson Ratio"]
    return


def Required_Planet_Axle_Shear(applied_moment):
    """
    Computes the required shear force applied on all edges of planet axle in order to induce the applied moment
    """
    Force_required_at_distance = applied_moment/(planet_axle_radius / 1000) #[N]
    Planet_Axle_Circumference = 2 * math.pi * planet_axle_radius   #[mm]
    Shear_per_unit_length = Force_required_at_distance/Planet_Axle_Circumference #[N/mm]

    return Shear_per_unit_length

def Preliminary_Calculations(mesh_name):
    global model_name, R_carrier_axle_adjusted, applied_shear, motion_table, moment_table, n_loadstep
    model_name = run_file[0:-5]
                #Ring mesh1  #Planet mesh1         #Crack length

    R_carrier_axle_adjusted = planet_carrier_pcr + move_planet_up  # Adjusted carrier axle height with similar effect as backlash

    applied_shear = Required_Planet_Axle_Shear(applied_moment)

    motion_table = "motion_" + str(n_increments)  # The rotation table to use for the analysis
    moment_table = "moment_" + str(n_increments)
    n_loadstep = n_increments*2 #+1 # Make sure there is a loadstep for each datapoint in the provided motion and moment tables


# Py_Mentat code
########################################################################################################################
def import_ring(bdf_file):
    # Prevents log from opening up
    py_send("*prog_option import_nastran:show_log:off") 
    
    # Import the planet mesh
    string = '*import nastran ' + '"' + '..\\Mesh\\' + bdf_file + '.bdf"'
    py_send(string)
    
    # Create a deformable contact body of the ring elements
    py_send("*add_contact_body_elements all_existing  *new_cbody mesh *contact_option state:solid *contact_option skip_structural:off ") 
    py_send("*add_contact_body_elements *contact_body_name Ring ") 
    py_send("*add_contact_body_elements all_existing ")

    py_send("*select_clear")
    py_send("*select_elements_cbody Ring")  # Select the planet elemets

    # Rotate Ring
    py_send("*prog_option move:mode:rotate")

    py_send("*set_move_centroid x 0")  # Set the point around which the ring gear should be rotating
    py_send("*set_move_centroid y 0")
    py_send("*set_move_centroid z 0")

    py_send("*set_move_rotation z " + str(ring_gear_rotation))
    py_send("*move_elements all_selected")


    # Create a geometrical contact body on the outside of the ring gear
    py_send("*set_curve_type circle_cr") #Curve type to circle
    ring_centre = "0 0 0 "
    ring_radius = str(ring_gear_external_radius)
    py_send("*add_curves " + ring_centre + ring_radius) # Draw the circle
    
    py_send("*new_cbody geometry *contact_option geometry_nodes:off ")
    py_send("*contact_body_name Housing")
    
    py_send("*contact_option control:position")
    py_send("*contact_value prot " + str(total_rotation))    #rotation total magnitude
    py_send("*cbody_param_table prot " + motion_table)  # Rotation table 
    
    py_send("*add_contact_body_curves all_existing")
    return


def import_planet(bdf_file):
    # Prevents log from opening up
    py_send("*prog_option import_nastran:show_log:off") 
    
    # Import the planet mesh
    string = '*import nastran ' + '"' + '..\\Mesh\\' + bdf_file + '.bdf"'
    py_send(string)
    
    py_send("*select_elements_cbody Ring") # Select existing contact body elements so that the new ones can be selected
    
    # Create a deformable contact body of the planet elements    
    py_send("*new_cbody mesh *contact_option state:solid *contact_option skip_structural:off")
    py_send("*contact_body_name Planet")
    py_send("*add_contact_body_elements all_unselected")
    
    py_send("*select_clear")
    py_send("*select_elements_cbody Planet") # Select the planet elemets
    
    # Move planet
    py_send("*prog_option move:mode:translate")
    py_send("*set_move_translation y " + str(move_planet_up))  # Move the planet in the y direction
    py_send("*move_elements all_selected")


    py_send("*prog_option move:mode:rotate")
    py_send("*set_move_centroid y " + str(R_carrier_axle_adjusted)) # Rotate the planet about this point
    py_send("*set_move_rotation z " + str(rotate_planet))
    py_send("*move_elements all_selected")

    
    # Create a geometrical contact body on the outside of the ring gear
    py_send("*add_nodes 0 " + str(R_carrier_axle_adjusted) + " 0") #Add the reference nodes for the contact body
    py_send("*add_nodes 0 " + str(R_carrier_axle_adjusted + 5) + " 0") #Add the reference nodes for the contact body
    
    py_send("*renumber_all") # Renumber so we can get the node number of the selected node
    n_nodes = py_get_int("nnodes()")  #Find out the number of nodes
    
    ID_Control_Node = int(n_nodes -1) # Set the node number of the control node and auxlilary nodes to be used 
    ID_Auxilary_Node = int(n_nodes)
    
    py_send("*set_curve_type circle_cr") #Curve type to circle
    planet_centre = "0 " + str(R_carrier_axle_adjusted) + " 0 "

    
    py_send("*add_curves " + planet_centre + str(planet_axle_radius)) # Draw the circle
    
    py_send("*new_cbody geometry *contact_option geometry_nodes:off ") # Make contact body
    py_send("*contact_body_name Carrier_Axle")  
    
    py_send("*contact_option control:load")
    py_send("*cbody_control_node " + str(ID_Control_Node)) # Set the primary control node
    py_send("*cbody_control_node_rot " + str(ID_Auxilary_Node)) # Set the auxilary control node

    py_send("*select_clear")
    py_send("*select_curves_cbody_all") #Select all curves associated with a contact body in order to ultimately select the new curve
    py_send("*add_contact_body_curves all_unselected")

    #Add a fixed displacement constraint to the Carrier axle contact body
    py_send("*new_apply *apply_type fixed_displacement") 
    py_send("*apply_name Carrier_Axle_Fix")
    py_send("*apply_dof x *apply_dof_value x")
    py_send("*apply_dof y *apply_dof_value y")
    py_send("*apply_dof z *apply_dof_value z")
    py_send("*apply_dof rx *apply_dof_value rx")
    py_send("*apply_dof ry *apply_dof_value ry")
    
    py_send("*add_apply_nodes " + str(ID_Control_Node) + " # | End of List")
            
    #Add moment to auxilary control node
    #py_send("*new_apply *apply_type point_load")
    #py_send("*apply_dof my *apply_dof_value my")
    #py_send("*apply_dof mx *apply_dof_value mx")
    #py_send("*apply_dof z *apply_dof_value z")
    #py_send("*apply_dof y *apply_dof_value y")
    #py_send("*apply_dof x *apply_dof_value x")
    #py_send("*apply_dof mz *apply_dof_value mz 1e2") #Set the moment magnitude here
    #py_send("*apply_dof_table mz " + moment_table) # Associate a table with this moment
    #py_send("*apply_name Planet_Torque")
    #py_send("*add_apply_nodes " + str(ID_Auxilary_Node) + " # | End of List")


    #Add a shear load to the planet inside to induce a moment
    py_send("*select_clear")
    py_send("*select_method_point_dist")
    py_send("*set_select_distance " + str(planet_axle_radius * 1.01))
    py_send("*select_edges 0.000000000000e+00   " + str(R_carrier_axle_adjusted) + "  0.000000000000e+00") #Select all nodes on the inside edge of the gear

    py_send("*new_apply *apply_type edge_load") 
    py_send("*apply_name Load_To_Induce_Moment") 
    py_send("*apply_dof su *apply_dof_value su " + str(applied_shear)) 
    py_send("*apply_dof_table su " + moment_table)
    py_send("*apply_option edge_load_mode:length")  # Set edge shear force specification to be unit length
    py_send("*add_apply_edges all_selected")

    py_send("*arrow_length 4") #Ensures the boundary conditions do not overpower the sketch
    py_send("*redraw")
    return


def contact():
    """Sets the contact interactions and contact table"""
    # Contact interaction glued 
    py_send("*new_interact mesh:geometry *interact_option state_1:solid") 
    py_send("*interact_name Glued_Interact")
    py_send("*interact_option contact_type:glue")
    
    # Contact interaction touch deformable on deformable
    py_send("*new_interact mesh:mesh *interact_option state_1:solid *interact_option state_2:solid")
    py_send("*interact_name Tooth_on_Tooth")
    py_send("*interact_param friction " + str(friction_coefficient))
    
    # Make a contact table
    py_send("*new_contact_table") #Creates a new contact table
    
    py_send("*ctable_entry Ring Housing")
    py_send("*contact_table_option Ring Housing contact:on") # Make glued connection between ring and housing
    py_send("*prog_string ctable:old_interact Glued_Interact *ctable_entry_interact Ring Housing")
    
    py_send("*ctable_entry Planet Carrier_Axle")
    py_send("*contact_table_option Planet Carrier_Axle contact:on") # Make glued connection between planet and carrier axle
    py_send("*prog_string ctable:old_interact Glued_Interact *ctable_entry_interact Planet Carrier_Axle")
    
    py_send("*ctable_entry Ring Planet")
    py_send("*contact_table_option Ring Planet contact:on") # Make glued connection between planet and carrier axle
    py_send("*prog_string ctable:old_interact Tooth_on_Tooth *ctable_entry_interact Ring Planet")

    return


def loadcase():
    """Sets up the loadcase"""
    py_send("*new_loadcase *loadcase_type struc:static") #Create a new loadcase
    py_send("*loadcase_ctable ctable1")  # Activate contact in the loadcase
    py_send("*loadcase_option nonpos:on") # Activate non positive definite for when gears are not touching

    py_send("*loadcase_value nsteps " + str(n_loadstep)) #Set the number of loadcase steps to be used

    py_send("*edit_loadcase lcase1")
    py_send("*loadcase_option converge:resid_and_disp")  # Set convergence to be based on displacements and residuals
    return 


def material_properties():
    """Sets the material properties, units are in mm"""
    
    py_send("*new_mater standard *mater_option general:state:solid *mater_option general:skip_structural:off")
    py_send("*mater_param structural:youngs_modulus " + str(E))
    py_send("*mater_param structural:poissons_ratio " + str(v))
    py_send("*mater_name Gear_Material")
    py_send("*add_mater_elements all_existing")
    return
    

def geometrical_properties_and_element_types():
    """Sets the geometrical properties and element types so that a 2D analysis can be performed"""
    
    py_send("*new_geometry *geometry_type mech_planar_pstress") # New plane stress geometry

    py_send("*geometry_param norm_to_plane_thick " + str(gear_thickness)) # Set the thickness of the geometry
    py_send("*geometry_option assumedstrn:on") # Assumed strain active
    py_send("*add_geometry_elements all_existing") #Add this property to all of the geometry
    
    #Element types
    #py_send("*element_type 124 all_existing") #Change all of the elements to plane stress full integration second order
    py_send("*element_type 3 all_existing") # Plane stress full integration quad 1st order
    py_send("*element_type 201 all_existing") # Plane stress full integration tri 1st order
    
    #Flip elements to the appropriate orientation
    py_send("*check_upside_down") # Find and select the upside-down elements
    py_send("*flip_elements all_selected") # Flip the elements that were upside-down
    return


def job(name):
    py_send("*new_job structural") # Start a new job
    py_send("*add_job_loadcases lcase1")  # Use loadcase 1
    py_send("*job_option strain:large")  # Use large strain formulation
    py_send("*job_option dimen:pstress")  # Plane stress
    py_send("*job_contact_table ctable1")  # Set up initial contact to be contact table 1
    py_send("*job_option follow:on")  # Enables follower force
    py_send("*job_option friction_model:coul_stick_slip")  # Use Stick slip friction model

    #Solver
    py_send("*update_job")
    py_send("*job_option solver:pardiso")  # Use Pardiso Solver
    #py_send("*job_option assem_recov_multi_threading:on")
    #py_send("*job_param assem_recov_nthreads 8")  # Use multiple threads for assembly and recovery
    #py_send("*job_option pardiso_multi_threading:on")
    #py_send("*job_param nthreads 8")  # Use multiple threads for solution

    #Job results
    py_send("*add_post_var von_mises") # Add equivalent von mises stress

    #Run the Job
    py_send("*update_job")
    py_send("*save_model")
    py_send("*submit_job 1 *monitor_job")
    #print("sleep_start")
    #time.sleep(20)
    #print("sleep_end")
    
    #py_send("*update_job")
    #py_send("*post_open_default") #Open the default post file
    #py_send("*post_monitor")

    run = True
    t_prev = 99
    t_start = time.time()
    # Check if file is being modified
    while run == True:
        fname = "..\\Run_Dir\\" + model_name + "_job1" + ".sts"

        time.sleep(5)  # Check every 10 seconds if the simulation is done running
        t = os.path.getmtime(fname)


        if t_prev == t:
            run = False
            print("Done with mesh ", name," in ",time.time()-t_start, " seconds")
        else:
            t_prev = t

    return


def tables():
    """Creates tables from generated text files"""
    
    py_send('*md_table_read_any "' + "..\\Tables\\" + motion_table + '.txt"') #Imports the motion table
    py_send('*set_md_table_type 1 time') # X axis of table is time
    
    py_send('*md_table_read_any "' + "..\\Tables\\" + moment_table + '.txt"') #Imports the moment table
    py_send('*set_md_table_type 1 time') # X axis of table is time
    return()

def create_model():
    py_send("*new_model yes") # Start a new model without saving
    py_send('*set_save_formatted off *save_as_model "' + str(model_name) + '.mud" yes')
    py_send("*select_clear")
    return

def run_Post_Proc_TVMS():
    py_send("*py_file_run Post_Proc_TVMS.py")

def main():

    load_run_file()

    for crack_length in range(1):

        name = "m1_a" + str(crack_length) + "mm"
        Preliminary_Calculations(name)

        create_model()

        tables()

        import_ring(ring_mesh)

        import_planet(name)

        contact()

        loadcase()

        geometrical_properties_and_element_types()

        material_properties()

        job(name)

        #run_Post_Proc_TVMS()

    return 

if __name__ == '__main__' :
    py_connect("", 40007)
    main()
    py_disconnect()
