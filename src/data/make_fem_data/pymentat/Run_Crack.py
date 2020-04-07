from py_mentat import *
import time
import math

# Simulation Parameters
########################################################################################################################


planet_bdf = "Planet_Marc_Mesh_0311.bdf"

model_name = "crack_script_dev"



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
E = 200000  # MPa
v = 0.3

def import_planet(bdf_file):
    # Prevents log from opening up
    py_send("*prog_option import_nastran:show_log:off")

    # Import the planet mesh
    string = '*import nastran ' + '"' + '..\\Mesh\\' + bdf_file + '"'
    py_send(string)

    # Create a deformable contact body of the planet elements
    py_send("*new_cbody mesh *contact_option state:solid *contact_option skip_structural:off")
    py_send("*contact_body_name Planet")
    py_send("*add_contact_body_elements all_existing")


    # Add a fixed displacement constraint

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
    py_send("*crack_param crack_growth_incr " + str(Maximum_Crack_Growth_Increment))
    py_send("*crack_option high_cycle_fatigue:on") # Make use of high cycle fatigue calculation
    py_send("*crack_option high_cycle_fatigue_meth:paris") # Use Paris Law
    py_send("*crack_option crack_growth_scale_meth:constant")  # Sets scale method to be constant rather than exponential or fatigue law
    py_send("*crack_option crack_growth_incr_meth:strs_int")  # Use stress intensity fator as basis for the paris law
    py_send("*crack_option crack_growth_paris_form:basic") # Make use of the basic paris law. Not square root Paris law
    py_send("*crack_param paris_law_threshold_k " + str(Paris_Law_Threshold)) # Set the Paris Law threashold to be zero
    py_send("*crack_param paris_law_c_k " + str(Paris_Law_C))

    py_send("*crack_param paris_law_m_k " + str(Paris_Law_m))
    py_send("*crack_param min_growth_incr_k " + str(Minimum_Growth_Increment))


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
    py_send("*select_nodes 0.000000000000e+00   " + str(R_carrier_axle_adjusted) + "  0.000000000000e+00") #Select all nodes on the inside edge of the gear

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
    py_send("*apply_dof p *apply_dof_value p " + str(Applied_Load))
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

    py_send("*loadcase_value time " + str(time))  # Set the number of loadcase steps to be used
    py_send("*loadcase_value nsteps " + str(n_steps))  # Set the number of loadcase steps to be used

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

    py_send("*new_geometry *geometry_type mech_three_shell")  # New plane stress geometry


    py_send("*geometry_param thick " + str(gear_thickness))  # Set the thickness of the geometry
    py_send("*add_geometry_elements all_existing")  # Add this property to all of the geometry

    # Element types
    # py_send("*element_type 124 all_existing") #Change all of the elements to plane stress full integration second order
    #  py_send("*element_type 3 all_existing")  # Plane stress full integration quad 1st order
    py_send("*element_type 201 all_existing")  # Plane stress full integration tri 1st order

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
    py_send("*add_post_var von_mises")  # Add equivalent von mises stress

    # Run the Job
    py_send("*update_job")
    py_send("*save_model")
    py_send("*submit_job 1 *monitor_job")
    # print("sleep_start")
    # time.sleep(20)
    # print("sleep_end")

    # py_send("*update_job")
    # py_send("*post_open_default") #Open the default post file
    # py_send("*post_monitor")
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
    py_send('*set_save_formatted off *save_as_model "' + str(model_name) + '.mud" yes')
    py_send("*select_clear")
    return


def main():
    create_model()

    tables()

    #import_ring(ring_mesh)

    import_planet(planet_bdf)

    #contact()

    #loadcase()

    geometrical_properties_and_element_types()

    material_properties()

    crack_init()

    boundary_conditions()

    loadcase()

    job()

    return


if __name__ == '__main__':
    py_connect("", 40007)
    main()
    py_disconnect()
