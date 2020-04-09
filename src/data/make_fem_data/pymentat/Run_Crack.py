from py_mentat import *
from py_post import *
import time
import json
import os

# Simulation Parameters
########################################################################################################################


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
        ring_mesh, planet_mesh,\
        n_increments,\
        applied_load,\
        friction_coefficient,\
        gear_thickness,\
        move_planet_up ,\
        rotate_planet,\
        planet_carrier_pcr,\
        ring_gear_external_radius,\
        ring_gear_rotation,\
        planet_axle_radius,\
        E,\
        v,\
        crack_start_coord ,\
        crack_end_coord,\
        fatigue_time_period ,\
        maximum_crack_growth_increment , \
        minimum_crack_growth_increment, \
        paris_law_threshold ,\
        paris_law_C ,\
        paris_law_m,\
        time



    file_paths() # Sets global variables for file paths

    run_file = py_get_string("run_file")
    run_file_path = run_input_files_dir + "\\" + run_file

    print(" ")
    print(" ")
    print("Begin crack simulation, run file: ", run_file)

    # Open the input file
    with open(run_file_path) as json_file:
        input = json.load(json_file)

        # Simulation Parameters
        ########################################################################################################################
        # Mesh
        planet_mesh = input["Mesh"]["Planet Mesh"]

        # Loadcase

        n_increments = input["Crack Properties"]["Load Case"]["Number Of Loadsteps"]  # int , Uneven number
        # Loadcase
        #time = n_increments  # Set the number of steps to be the same as the time

        applied_load = input["Crack Properties"]["Load Case"]["Applied Load"]  # Cyclic edge force maximum magnitude [N/m]

        # Geometry
        gear_thickness = input["Geometry"]["Gear Thickness"]  # [mm]

        rotate_planet = input["Crack Properties"]["Position Adjustment"]["Rotate Planet"]  # Angle the planet gear should be rotated [degrees]
        planet_carrier_pcr = input["Geometry"]["Planet Carrier Pitch Centre Radius"]  # Pitch Centre Radius of planet carrier axle

        planet_axle_radius = input["Geometry"]["Planet Axle Radius"]  # Internal radius of the planet gear [mm]

        # Material Parameters
        ##############################################################################################################
        E = input["Material"]["Young's Modulus"] # MPa
        v = input["Material"]["Poisson Ratio"]

        # Crack
        crack_start_coord = input["Crack Properties"]["Crack Initialization"]["Cut Start Coordinate"]  # x,y coordinate of crack initiator start
        crack_end_coord = input["Crack Properties"]["Crack Initialization"]["Cut End Coordinate"]  # x,y coordinate of crack initiator end

        fatigue_time_period = input["Crack Properties"]["Crack Growth Parameters"]["Fatigue Time Period"] #[s]
        maximum_crack_growth_increment = input["Crack Properties"]["Crack Growth Parameters"]["Maximum Crack Growth Increment"] #[mm]
        minimum_crack_growth_increment = input["Crack Properties"]["Crack Growth Parameters"]["Minimum Crack Growth Increment"]  # [mm]
        paris_law_threshold = input["Crack Properties"]["Crack Growth Parameters"]["Maximum Crack Growth Increment"]  # [MPa sqrt(mm)]
        paris_law_C = input["Crack Properties"]["Crack Growth Parameters"]["Paris Law C"]  # [m/(cycle*MPa m^0.5)]
        paris_law_m = input["Crack Properties"]["Crack Growth Parameters"]["Paris Law m"]
    return


def import_planet(bdf_file):
    # Prevents log from opening up
    py_send("*prog_option import_nastran:show_log:off")

    # Import the planet mesh
    mesh = '*import nastran ' + '"' + mesh_dir + "\\" + planet_mesh + '"'
    py_send(mesh)

    # Create a deformable contact body of the planet elements
    py_send("*new_cbody mesh *contact_option state:solid *contact_option skip_structural:off")
    py_send("*contact_body_name Planet")
    py_send("*add_contact_body_elements all_existing")


    py_send("*prog_option move:mode:rotate")
    py_send("*set_move_centroid y " + str(planet_carrier_pcr)) # Rotate the planet about this point
    py_send("*set_move_rotation z " + str(rotate_planet))
    py_send("*move_elements all_existing")
    return

def crack_init():
    #  Construct the line that will act as crack initiator
    py_send("*add_points " + str(crack_start_coord[0]) + "   " + str(crack_start_coord[1]) + "   0")
    py_send("*add_points " + str(crack_end_coord[0]) + "   " + str(crack_end_coord[1]) + "   0")

    py_send("*set_curve_type line")
    py_send("*add_curves 1 2")



    #  Create Crack
    # Create a 2D crack that relies on VCCT
    py_send("*new_crack *crack_option dimension:2d *crack_option application:vcct")
    py_send("*crack_name Root_Crack")
    py_send("*crack_option usage:template_only")
    py_send("*crack_option crack_prop_mode:fatigue")
    py_send("*crack_option crack_growth_method:cut_through_elements") # Using cut trough elements instead of adaptive remeshing
    py_send("*crack_option crack_growth_dir:max_hoop_stress")  # Let crack grow in the direction of maximum hoop stress
    py_send("*crack_param fatigue_time_period " + str(fatigue_time_period)) # Time over which the stress intensities for growth is calculated
    py_send("*crack_option crack_growth_incr:scaled")  # User defined, Allows you to use high cycle fatigue
    py_send("*crack_param crack_growth_incr " + str(maximum_crack_growth_increment))
    py_send("*crack_option high_cycle_fatigue:on") # Make use of high cycle fatigue calculation
    py_send("*crack_option high_cycle_fatigue_meth:paris") # Use Paris Law
    py_send("*crack_option crack_growth_scale_meth:constant")  # Sets scale method to be constant rather than exponential or fatigue law
    py_send("*crack_option crack_growth_incr_meth:strs_int")  # Use stress intensity fator as basis for the paris law
    py_send("*crack_option crack_growth_paris_form:basic") # Make use of the basic paris law. Not square root Paris law
    py_send("*crack_param paris_law_threshold_k " + str(paris_law_threshold)) # Set the Paris Law threashold to be zero
    py_send("*crack_param paris_law_c_k " + str(paris_law_C))

    py_send("*crack_param paris_law_m_k " + str(paris_law_m))
    py_send("*crack_param min_growth_incr_k " + str(minimum_crack_growth_increment))


    #  Add the crack initiator
    py_send("new_crackinit *crackinit_option dimension:2d *crackinit_option insert_method:cut_mesh") # 2D Mesh cutting crack initiator
    py_send("*crackinit_name Max_Bending_Stress_Crack_Init")
    py_send("*crackinit_template_crack Root_Crack")
    py_send("*add_crackinit_curves all_existing")

    return

def boundary_conditions():
    # Add the fixed displacement
    py_send("*select_clear")
    py_send("*select_method_point_dist")
    py_send("*set_select_distance " + str(planet_axle_radius * 1.01))
    py_send("*select_nodes 0.000000000000e+00   " + str(planet_carrier_pcr) + "  0.000000000000e+00") #Select all nodes on the inside edge of the gear

    py_send("*new_apply *apply_type fixed_displacement")
    py_send("*apply_name Fixed_Axle")

    py_send("*apply_dof x *apply_dof_value x")
    py_send("*apply_dof y *apply_dof_value y")
    py_send("*apply_dof z *apply_dof_value z")
    py_send("*apply_dof rx *apply_dof_value rx")
    py_send("*apply_dof ry *apply_dof_value ry")
    py_send("*apply_dof rz *apply_dof_value rz")

    py_send("*add_apply_nodes all_selected")

    py_send("*arrow_length 2")  # Ensures the boundary conditions do not overpower the sketch
    py_send("*redraw")



    #  Add the edge load
    py_send("*select_clear")
    py_send("*select_method_point_dist")
    py_send("*set_select_distance " + str(0.8))
    py_send("*select_edges 1.4   72.5   0.000000000000e+00") #Select all nodes on the inside edge of the gear

    py_send("*new_apply *apply_type edge_load")
    py_send("*apply_name Meshing_Force")
    py_send("*apply_dof p *apply_dof_value p " + str(applied_load))
    py_send("*apply_dof_table p table1")
    py_send("*add_apply_edges all_selected")

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
    py_send("*interact_param friction " + str(Friction_Coefficient))

    # Make a contact table
    py_send("*new_contact_table")  # Creates a new contact table

    py_send("*ctable_entry Ring Housing")
    py_send("*contact_table_option Ring Housing contact:on")  # Make glued connection between ring and housing
    py_send("*prog_string ctable:old_interact Glued_Interact *ctable_entry_interact Ring Housing")

    py_send("*ctable_entry Planet Carrier_Axle")
    py_send(
        "*contact_table_option Planet Carrier_Axle contact:on")  # Make glued connection between planet and carrier axle
    py_send("*prog_string ctable:old_interact Glued_Interact *ctable_entry_interact Planet Carrier_Axle")

    py_send("*ctable_entry Ring Planet")
    py_send("*contact_table_option Ring Planet contact:on")  # Make glued connection between planet and carrier axle
    py_send("*prog_string ctable:old_interact Tooth_on_Tooth *ctable_entry_interact Ring Planet")

    return


def loadcase():
    """Sets up the loadcase"""
    py_send("*new_loadcase *loadcase_type struc:static")  # Create a new loadcase

    py_send("*loadcase_value time " + str(n_increments))  # Set the number of loadcase steps to be used
    py_send("*loadcase_value nsteps " + str(n_increments))  # Set the number of loadcase steps to be used

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

    py_send("*new_geometry *geometry_type mech_three_shell")  # New shell geometry


    py_send("*geometry_param thick " + str(gear_thickness))  # Set the thickness of the geometry
    py_send("*add_geometry_elements all_existing")  # Add this property to all of the geometry

    # Element types
    # py_send("*element_type 124 all_existing") #Change all of the elements to plane stress full integration second order
    #  py_send("*element_type 3 all_existing")  # Plane stress full integration quad 1st order
    #py_send("*element_type 201 all_existing")  # Plane stress full integration tri 1st order
    #py_send("*element_type 26 all_existing") # Plane stress full integration quad 2nd order
    #py_send("*element_type 124 all_existing") # Plane stress full integration tri 2nd order
    #py_send("*element_type 139 all_existing")
    #py_send("*element_type 49 all_existing") # Second order tri thin shell


    # Change tri elements to collapsed quads so that crack propagation by mesh splitting is possible
    py_send("*set_change_class quad8")
    py_send("*change_elements_class all_existing")

    py_send("*element_type 22 all_existing") # Second order quad thick shell

    # Flip elements to the appropriate orientation
    py_send("*check_upside_down")  # Find and select the upside-down elements
    py_send("*flip_elements all_selected")  # Flip the elements that were upside-down
    return


def job():
    py_send("*new_job structural")  # Start a new job
    py_send("*add_job_loadcases lcase1")  # Use loadcase 1
    py_send("*job_option strain:large")  # Use large strain formulation

    py_send("*remove_job_applys Meshing_Force") # Ensure that the only initial boundary condition is the fixed displacement

    py_send("*add_job_crackinit Max_Bending_Stress_Crack_Init")  # Add the crack init


    # Solver
    py_send("*update_job")
    py_send("*job_option solver:pardiso")  # Use Pardiso Solver
    #py_send("*job_option assem_recov_multi_threading:on")
    #py_send("*job_param assem_recov_nthreads 8")  # Use multiple threads for assembly and recovery
    #py_send("*job_option pardiso_multi_threading:on")
    #py_send("*job_param nthreads 8")  # Use multiple threads for solution

    # Job results
    #py_send("*add_post_var von_mises")  # Add equivalent von mises stress

    # Run the Job
    py_send("*update_job")
    py_send("*save_model")
    py_send("*submit_job 1 *monitor_job")


    run = True
    t_prev = 99
    t_start = time.time()
    # Check if file is being modified
    while run == True:
        fname = run_dir_dir + "\\" + run_file[0:-5] + "_crack" + "_job1" + ".sts"

        time.sleep(5)  # Check every 5 seconds if the simulation is done running
        t = os.path.getmtime(fname)

        if t_prev == t:
            run = False
            print("Done with crack simulation ", run_file[0:-5]," in ", time.time()-t_start, " seconds")
        else:
            t_prev = t

    return



def tables():
    """Creates tables from generated text files"""

    # Make a table that completes 5 cycles in the 2 second fatigue period. This is the time over which the stress intensity is determined?
    py_send("*new_md_table 1 1")
    py_send("*set_md_table_type 1 time")
    py_send("*set_md_table_max_v 1 10")
    py_send("*table_add 0 0")
    py_send("*table_add 2 0")
    py_send("*table_add 4 0")
    py_send("*table_add 6 0")
    py_send("*table_add 8 0")
    py_send("*table_add 10 0")
    py_send("*table_add 1 1")
    py_send("*table_add 3 1")
    py_send("*table_add 5 1")
    py_send("*table_add 7 1")
    py_send("*table_add 9 1")
    return


def create_model():
    py_send("*new_model yes")  # Start a new model without saving
    py_send('*set_save_formatted off *save_as_model "' + run_file[0:-5] + '_crack' + '.mud" yes')
    py_send("*select_clear")
    return


name = 'crack_script_dev_job1'
n_increm = 11

def open_file():
    py_send("*post_open " + run_dir_dir + "\\" + run_file[0:-5] + '_crack' + "_job1.t16")
    py_send("*post_next")
    py_send("*fill_view")
    py_send("*zoom_box")
    py_send("*zoom_box(2,0.472408,0.085288,0.525084,0.149254)")
    return

def extract_meshes():
    p = post_open(run_dir_dir + "\\" + run_file[0:-5] + '_crack' + "_job1.t16")
    ninc = p.increments()
    p.moveto(ninc - 1)
    n = p.global_values()
    #for i in range(0, n):
    #    print(p.global_value_label(i),p.global_value(i))


    mesh_extract = True
    new_dir = mesh_dir + "\\" + run_file[0:-5] + "_planet_meshes"


    try:
        # Create target Directory
        os.mkdir(new_dir)

    except FileExistsError:
        print("run file ", run_file[0:-5], " already ran: Overwriting")


    for i in range(n_increm):
        ninc = p.increments()
        p.moveto(ninc - i)
        py_send("*post_next")
        if mesh_extract == True:
            crack_length = p.global_value(n-3)
            print(p.global_value_label(n-3), crack_length)

            py_send("*export nastran '" + new_dir + "\\" + str(crack_length) + "mm.bdf' yes")

        mesh_extract = not mesh_extract
    return



def post():
    #py_send("*post_close")

    open_file()

    extract_meshes()



def main():
    load_run_file()

    create_model()

    tables()

    import_planet(planet_mesh)

    geometrical_properties_and_element_types()

    material_properties()

    crack_init()

    boundary_conditions()

    loadcase()

    job()

    post()

    return


if __name__ == '__main__':
    py_connect("", 40007)
    main()
    py_disconnect()
