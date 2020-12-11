from datetime import time

from py_mentat import *
import time
import math
import os
import json


def file_paths():
    """
    Manages file paths
    """
    global run_dir_dir, mesh_dir, run_input_files_dir, tables_dir,fem_data_raw_dir
    #Imports with pyMentat are finicky. For that reason, definitions.py cannot be imported and the code below is required.
    pymentat_dir = os.path.dirname(os.path.abspath(__file__))
    make_fem_data_dir = os.path.dirname(pymentat_dir)
    src_data_dir = os.path.dirname(make_fem_data_dir)
    src_dir = os.path.dirname(src_data_dir)
    root_dir = os.path.dirname(src_dir)

    run_dir_dir = os.path.join(root_dir,"models\\fem\\run_dir")
    mesh_dir = os.path.join(root_dir, "models\\fem\\mesh")
    tables_dir = os.path.join(root_dir, "models\\fem\\tables")
    run_input_files_dir = os.path.join(root_dir, "models\\fem\\run_input_files")

    fem_data_raw_dir = os.path.join(root_dir,"data\\external\\fem\\raw")
    return


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

    file_paths() # Sets global variables for file paths

    run_file = py_get_string("run_file")
    run_file_path = run_input_files_dir + "\\" + run_file

    print(" ")
    print(" ")
    print("Begin simulation, run file: ", run_file)

    # Open the input file
    with open(run_file_path) as json_file:
        input = json.load(json_file)

        # Simulation Parameters
        ################################################################################################################
        # Mesh
        ring_mesh = input["Mesh"]["Ring Mesh"]
        planet_meshes = mesh_dir + "\\" + run_file[0:-5] + "_" + "planet_meshes"

        if os.path.isdir(planet_meshes) == False:
            raise FileNotFoundError("Crack simulation for " + run_file + " not yet run.")

        # Loadcase
        total_rotation = input["TVMS Properties"]["Load Case"]["Total Rotation"] #The total angular distance rotated [rad]
        n_increments = input["TVMS Properties"]["Load Case"]["Number Of Loadsteps"] #int
        applied_moment = input["TVMS Properties"]["Load Case"]["Applied Moment"]  # Moment on planet gear [Nm]

        # Contact
        friction_coefficient = input["TVMS Properties"]["Contact"]["Friction Coefficient"]  # Dynamic friction coefficient for lubricated Cast iron on Cast iron https://www.engineeringtoolbox.com/friction-coefficients-d_778.html

        # Geometry
        gear_thickness =input["Geometry"]["Gear Thickness"] # [mm]

        move_planet_up = input["TVMS Properties"]["Position Adjustment"]["Move Planet Up"]#-1.65  # Distance the planet gear should be moved up [mm]
        rotate_planet = input["TVMS Properties"]["Position Adjustment"]["Rotate Planet"]#-1.72 - (360 / 24) * 2  # Angle the planet gear should be rotated [degrees]
        planet_carrier_pcr = input["Geometry"]["Planet Carrier Pitch Centre Radius"]  # Pitch Centre Radius of planet carrier axle

        ring_gear_external_radius = input["Geometry"]["Ring Gear External Radius"]  # External Radius of Ring gear [mm]
        ring_gear_rotation = input["TVMS Properties"]["Position Adjustment"]["Ring Gear Rotation"] #-(360 / 62) * 2  # Angle the ring gear should be rotated

        planet_axle_radius = input["Geometry"]["Planet Axle Radius"]  # Internal radius of the planet gear [mm]

        # Material Parameters
        ##############################################################################################################
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

def Preliminary_Calculations():
    global model_name, R_carrier_axle_adjusted, applied_shear, motion_table, moment_table, n_loadstep
    model_name = run_file[0:-5]

    R_carrier_axle_adjusted = planet_carrier_pcr + move_planet_up  # Adjusted carrier axle height with similar effect as backlash

    applied_shear = Required_Planet_Axle_Shear(applied_moment)

    motion_table = "motion_" + str(n_increments)  # The rotation table to use for the analysis
    moment_table = "moment_" + str(n_increments)
    n_loadstep = n_increments*2 #+1 # Make sure there is a loadstep for each datapoint in the provided motion and moment tables

# Py_Mentat TVMS code
########################################################################################################################
def import_ring(bdf_file):
    # Prevents log from opening up
    py_send("*prog_option import_nastran:show_log:off") 

    # Import the planet mesh
    planet_mesh = '*import nastran ' + '"' + mesh_dir + "\\" + bdf_file + '"'

    py_send(planet_mesh)
    
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
    planet_mesh = '*import nastran ' + '"' + planet_meshes + "\\" + bdf_file + '"'
    py_send(planet_mesh)
    
    py_send("*select_elements_cbody Ring") # Select existing contact body elements so that the new ones can be selected
    
    # Create a deformable contact body of the planet elements    
    py_send("*new_cbody mesh *contact_option state:solid *contact_option skip_structural:off")
    py_send("*contact_body_name Planet")
    py_send("*add_contact_body_elements all_unselected")
    
    py_send("*select_clear")
    py_send("*select_elements_cbody Planet") # Select the planet elemets
    
    # Make small adjustments to planet to ensure that there is no interference between meshes
    py_send("*prog_option move:mode:translate")
    # py_send("*set_move_translation y " + str(move_planet_up))  # Small planet adjustment in the y direction
    py_send("*set_move_translation y " + str(R_carrier_axle_adjusted))  # Small planet adjustment in the y direction
    py_send("*move_elements all_selected")


    py_send("*prog_option move:mode:rotate")
    py_send("*set_move_centroid y " + str(R_carrier_axle_adjusted)) # Rotate the planet about this point
    py_send("*set_move_rotation z " + str(rotate_planet))
    py_send("*move_elements all_selected")


    py_send("*add_nodes 0 " + str(R_carrier_axle_adjusted) + " 0") #Add the reference nodes for the contact body
    py_send("*add_nodes 0 " + str(R_carrier_axle_adjusted + 5) + " 0") #Add the reference nodes for the contact body
    
    py_send("*renumber_all") # Renumber so we can get the node number of the selected node
    n_nodes = py_get_int("nnodes()")  #Find out the number of nodes
    
    ID_Control_Node = int(n_nodes -1) # Set the node number of the control node and auxlilary nodes to be used 
    ID_Auxilary_Node = int(n_nodes)

    # Planet carrier axle that carries the planet gear
    py_send("*set_curve_type circle_cr")  # Curve type to circle
    planet_centre = "0 " + str(R_carrier_axle_adjusted) + " 0 "
    py_send("*add_curves " + planet_centre + str(planet_axle_radius)) # Draw the circle
    
    py_send("*new_cbody geometry *contact_option geometry_nodes:off ")  # Make contact body
    py_send("*contact_body_name Carrier_Axle")  
    
    py_send("*contact_option control:load")
    py_send("*contact_option load_control_rotation:allowed")  # Allow rotation of the contact body
    py_send("*cbody_control_node " + str(ID_Control_Node))  # Set the primary control node
    py_send("*cbody_control_node_rot " + str(ID_Auxilary_Node))  # Set the auxilary control node

    py_send("*select_clear")
    py_send("*select_curves_cbody_all") #Select all curves associated with a contact body in order to ultimately select the new curve
    py_send("*add_contact_body_curves all_unselected")

    py_send("*set_plot_curve_div_high") # Set the curve tolerance to be higher to let circles appear more circular
    py_send("*redraw")

    #Add a fixed displacement constraint to the Carrier axle contact body
    py_send("*new_apply *apply_type fixed_displacement") 
    py_send("*apply_name Carrier_Axle_Fix")
    py_send("*apply_dof x *apply_dof_value x")
    py_send("*apply_dof y *apply_dof_value y")
    # py_send("*apply_dof z *apply_dof_value z") # These are not constrained becuase in planar analysis third DOF would be rotation
    # py_send("*apply_dof rx *apply_dof_value rx")
    # py_send("*apply_dof ry *apply_dof_value ry")
    
    py_send("*add_apply_nodes " + str(ID_Control_Node) + " # | End of List")

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

    py_send("*arrow_length 2") #Ensures the boundary conditions do not overpower the sketch
    py_send("*redraw")

    py_send("*fill_view")
    return


def contact():
    """Sets the contact interactions and contact table"""
    # Contact interaction glued (rigid body on mesh)
    py_send("*new_interact mesh:geometry *interact_option state_1:solid") 
    py_send("*interact_name Glued_Interact")
    py_send("*interact_option contact_type:glue")
    py_send("*interact_option project_stress_free:on") # Turn stress free projection on to prevent stresses in planet due to contact with planet axle
    py_send("*interact_option dist_tol:redefined") # Redefine the contact tollerance to make sure there is contact between Planet_Axle and Planet
    py_send("*interact_param dist_tol 1")
    
    # Contact interaction touch deformable on deformable (tooth on tooth)
    py_send("*new_interact mesh:mesh *interact_option state_1:solid *interact_option state_2:solid")
    py_send("*interact_name Tooth_on_Tooth")
    py_send("*interact_param friction " + str(friction_coefficient))
    py_send("*interact_option delay_slide_off:on")  # Tangential contact extension, could prevent noisy TVMS readings
    
    # Make a contact table
    py_send("*new_contact_table") #Creates a new contact table
    
    py_send("*ctable_entry Ring Housing")
    py_send("*contact_table_option Ring Housing contact:on") # Make glued connection between ring and housing
    py_send("*prog_string ctable:old_interact Glued_Interact *ctable_entry_interact Ring Housing")
    
    py_send("*ctable_entry Planet Carrier_Axle")
    py_send("*contact_table_option Planet Carrier_Axle contact:on") # Make glued connection between planet and carrier axle
    py_send("*prog_string ctable:old_interact Glued_Interact *ctable_entry_interact Planet Carrier_Axle")
    
    py_send("*ctable_entry Ring Planet")
    py_send("*contact_table_option Ring Planet contact:on")
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
    
    #Element types (This has to be changed depending on whether input mesh is first or second order elements
    py_send("*element_type 124 all_existing") #Change all of the elements to plane stress full integration second order
    #py_send("*element_type 3 all_existing") # Plane stress full integration quad 1st order
    #py_send("*element_type 201 all_existing") # Plane stress full integration tri 1st order
    py_send("*element_type 26 all_existing") # Plane stress full integration quad 2nd order
    py_send("*element_type 124 all_existing") # Plane stress full integration tri 2ndorder
    
    #Flip elements to the appropriate orientation
    py_send("*check_upside_down") # Find and select the upside-down elements
    py_send("*flip_elements all_selected") # Flip the elements that were upside-down
    return


def job(mesh_name):
    py_send("*new_job structural") # Start a new job
    py_send("*add_job_loadcases lcase1")  # Use loadcase 1
    py_send("*job_option strain:large")  # Use large strain formulation

    #py_send("*job_option dimen:pstress")  # Select between plane stress and plane strain
    py_send("*job_option dimen:pstrain")  #

    py_send("*job_contact_table ctable1")  # Set up initial contact to be contact table 1
    py_send("*job_option follow:on")  # Enables follower force, This is required for the applied shear load
    py_send("*job_option friction_model:coul_stick_slip")  # Use Stick slip friction model
    #py_send("*job_option contact_method: node_segment") # Use node to segnment contact
    #py_send("*job_option friction_model:coulomb_bilinear") # Use bilinear coulomb (less accurate than stick slip)



    #Solver
    py_send("*update_job")
    py_send("*job_option solver:pardiso")  # Use Pardiso Solver
    py_send("*job_option assem_recov_multi_threading:on")
    py_send("*job_param assem_recov_nthreads 8")  # Use multiple threads for assembly and recovery
    py_send("*job_option pardiso_multi_threading:on")
    py_send("*job_param nthreads 8")  # Use multiple threads for solution

    #Job results
    py_send("*add_post_var von_mises") # Add equivalent von mises stress

    #Run the Job
    # py_send("*update_job")
    # py_send("*save_model")
    # py_send("*submit_job 1 *monitor_job")

    #print("sleep_start")
    #time.sleep(20)
    #print("sleep_end")
    
    #py_send("*update_job")
    #py_send("*post_open_default") #Open the default post file
    #py_send("*post_monitor")

    fname = run_dir_dir + "\\" + model_name + "_" + mesh_name[0:-4] + "_job1" + ".sts"
    run = True
    t_prev = os.path.getmtime(fname)
    t_start = time.time()
    # Check if file is being modified

    #print(fname)
    #os.remove(fname)  # Make the file does not exist so it does not write out results due to previous simulations, Permission error?
    # while run == True:
    #
    #
    #     time.sleep(100)  # Check every 20 seconds if the simulation is done running
    #                     # Make sure this is longer than it takes the a single increment of the simulation to run
    #     t = os.path.getmtime(fname)
    #
    #     if t_prev == t:
    #         run = False
    #         print("Done with mesh ", model_name + " :" + mesh_name," in < ",(time.time()-t_start)/60, " min")
    #     else:
    #         t_prev = t
            #print(t_prev)

    return

def tables():
    """Creates marc tables from generated text files"""

    if os.path.isfile(tables_dir + "\\" + motion_table + '.txt') == False:
        raise FileNotFoundError("Tables for selected number of load-steps not generated: Generate using src/data/make_fem_data/make_marc_table_data")

    py_send('*md_table_read_any "' + tables_dir + "\\" + motion_table + '.txt"') #Imports the motion table
    py_send('*set_md_table_type 1 time') # X axis of table is time
    
    py_send('*md_table_read_any "' + tables_dir + "\\" + moment_table + '.txt"') #Imports the moment table
    py_send('*set_md_table_type 1 time') # X axis of table is time
    return()

def create_model(planet_mesh):
    py_send("*new_model yes") # Start a new model without saving
    py_send('*set_save_formatted off *save_as_model "' + str(model_name) + "_" + planet_mesh[0:-4] + '.mud" yes')
    py_send("*select_clear")
    return


#  Post Processing the TVMS simulation
#####################################################################################
def open_file(planet_mesh):
    py_send("*post_open " + run_dir_dir + "\\" + model_name + "_" + planet_mesh[0:-4] + "_job1.t16")
    py_send("*post_next")
    py_send("*fill_view")

def angle_pos_history_plot(planet_mesh):
    time.sleep(5)  # Sleep for a while
    "Makes a history plot of the angular displacement of the planet axle rigid body"
    py_send("*history_collect 0 999999999 1")
    #py_send("*history_clear")
    py_send("*prog_option history_plot:data_carrier_type_x:global")
    py_send("*set_history_global_variable_x Time")

    py_send("*prog_option history_plot:data_carrier_type_y:cbody")  # This refers to y axis
    py_send("*set_history_data_carrier_cbody_y Carrier_Axle")
    py_send("*set_history_cbody_variable_y Angle Pos")

    py_send("*history_add_curve")
    py_send("*history_fit")

    py_send("*history_write " + fem_data_raw_dir + "\\" + run_file[0:-5] + "_" + planet_mesh[0:-4] + "_planet_angle" + ".txt yes")
    return


def post_processing(planet_mesh):
    time.sleep(5)  # Wait 5 seconds just in case
    #py_send("*post_close")
    open_file(planet_mesh)
    angle_pos_history_plot(planet_mesh)
    return

def main():

    load_run_file()

    for cracked_mesh in os.listdir(planet_meshes):
        if cracked_mesh.endswith('.bdf'):

            # Preliminary_Calculations()
            #
            # create_model(cracked_mesh)
            #
            # tables()
            #
            # import_ring(ring_mesh)
            #
            # import_planet(cracked_mesh)
            #
            # contact()
            #
            # loadcase()
            #
            # geometrical_properties_and_element_types()
            #
            # material_properties()
            #
            # job(cracked_mesh)

            post_processing(cracked_mesh)

    return 

if __name__ == '__main__' :
    py_connect("", 40007)
    main()
    py_disconnect()
