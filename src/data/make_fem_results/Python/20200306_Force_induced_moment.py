# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from py_mentat import *
import time




def import_ring(bdf_file):
    # Prevents log from opening up
    py_send("*prog_option import_nastran:show_log:off") 
    
    # Import the planet mesh
    string = '*import nastran ' + '"' + bdf_file + '"'
    #string = '*import ideas ' + '"' + bdf_file + '"'
    py_send(string)
    
    # Create a deformable contact body of the ring elements
    py_send("*add_contact_body_elements all_existing  *new_cbody mesh *contact_option state:solid *contact_option skip_structural:off ") 
    py_send("*add_contact_body_elements *contact_body_name Ring ") 
    py_send("*add_contact_body_elements all_existing ") 
    
    # Create a geometrical contact body on the outside of the ring gear
    py_send("*set_curve_type circle_cr") #Curve type to circle
    #ring_centre = "0 0 0.012 "
    ring_centre = "0 0 0 "
    ring_radius = "88.13"
    #ring_radius = "0.08813"
    #ring_radius = "0.16"
    py_send("*add_curves " + ring_centre + ring_radius) # Draw the circle
    
    py_send("*new_cbody geometry *contact_option geometry_nodes:off ")
    py_send("*contact_body_name Housing")
    
    total_rotation = 0.1
    py_send("*contact_option control:position")
    py_send("*contact_value prot " + str(total_rotation))    #rotation total magnitude
    py_send("*cbody_param_table prot motion_3")  # Rotation table 
    
    py_send("*add_contact_body_curves all_existing")
    return

def import_planet(bdf_file,move_y,move_rz):
    # Prevents log from opening up
    py_send("*prog_option import_nastran:show_log:off") 
    
    # Import the planet mesh
    string = '*import nastran ' + '"' + bdf_file + '"'
    #string = '*import ideas ' + '"' + bdf_file + '"'
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
    py_send("*set_move_translation y " + str(move_y))  # Move the planet in the y direction
    py_send("*move_elements all_selected")
    
    #R = 86.47/2000 + move_planet_up # Radius of planet carrier axle
    R = 86.47/2 + move_y # Radius of planet carrier axle
    print("R :", R)
    
    py_send("*prog_option move:mode:rotate")
    py_send("*set_move_centroid y " +str(R)) # Rotate the planet about this point
    py_send("*set_move_rotation z " +str(move_rz)) 
    py_send("*move_elements all_selected")

    
    # Create a geometrical contact body on the outside of the ring gear
    #py_send("*add_nodes 0 " + str(R) + " 0.012") #Add the reference nodes for the contact body
    #py_send("*add_nodes 0 " + str(R + 0.004) + " 0.012") #Add the reference nodes for the contact body
    
    #py_send("*add_nodes 0 " + str(R) + " 12") #Add the reference nodes for the contact body
    #py_send("*add_nodes 0 " + str(R + 5) + " 12") #Add the reference nodes for the contact body
    py_send("*add_nodes 0 " + str(R) + " 0") #Add the reference nodes for the contact body
    py_send("*add_nodes 0 " + str(R + 5) + " 0") #Add the reference nodes for the contact body
    
    py_send("*renumber_all") # Renumber so we can get the node number of the selected node
    n_nodes = py_get_int("nnodes()")  #Find out the number of nodes
    
    ID_Control_Node = int(n_nodes -1) # Set the node number of the control node and auxlilary nodes to be used 
    ID_Auxilary_Node = int(n_nodes)
    
    py_send("*set_curve_type circle_cr") #Curve type to circle
    #planet_centre = "0 " + str(R) + " 0.012 "
    planet_centre = "0 " + str(R) + " 0 "
    #planet_radius = str(29.32/2000)#
    planet_radius = 29.32/2#
    py_send("*add_curves " + planet_centre + str(planet_radius)) # Draw the circle
    
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
    py_send("*new_apply *apply_type point_load") 
    py_send("*apply_dof my *apply_dof_value my")
    py_send("*apply_dof mx *apply_dof_value mx")
    py_send("*apply_dof z *apply_dof_value z")
    py_send("*apply_dof y *apply_dof_value y")
    py_send("*apply_dof x *apply_dof_value x")
    py_send("*apply_dof mz *apply_dof_value mz 1e2") #Set the moment magnitude here
    py_send("*apply_dof_table mz moment_3") # Associate a table with this moment
    py_send("*apply_name Planet_Torque")
    py_send("*add_apply_nodes " + str(ID_Auxilary_Node) + " # | End of List")
            
            
    #Add a shear load to the planet inside to induce a moment
    planet_inside_bot = R - planet_radius
    py_send("*select_clear")
    py_send("*select_method_user_box")
    py_send("*select_edges -1 1 " + str(planet_inside_bot -0.1) + " "  + str(planet_inside_bot+0.1) + " -3 3")

    py_send("*new_apply *apply_type edge_load") 
    py_send("*apply_name Load_To_Induce_Moment") 
    py_send("*apply_dof su *apply_dof_value su 200") 
    py_send("*apply_dof_table su moment_3") 
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
    return 

def material_properties():
    """Sets the material properties, units are in mm"""
    E = 200 #MPa
    v = 0.3
    
    py_send("*new_mater standard *mater_option general:state:solid *mater_option general:skip_structural:off")
    py_send("*mater_param structural:youngs_modulus " + str(E))
    py_send("*mater_param structural:poissons_ratio " + str(v))
    py_send("*mater_name Gear_Material")
    py_send("*add_mater_elements all_existing")
    return
    

def geometrical_properties_and_element_types():
    """Sets the geometrical properties and element types so that a 2D analysis can be performed"""
    #Geometrical properties
    #py_send("*edit_geometry pshell-1 *remove_current_geometry") # Remove the geometrical properties as imported
    #py_send("*edit_geometry pshell-1_1 *remove_current_geometry") # Remove the geometrical properties as imported
    #The names above are generated by the mesher
    
    py_send("*new_geometry *geometry_type mech_planar_pstress") # New plane stress geometry
    
    #gear_thickness = " 0.012"
    gear_thickness = " 12"
    py_send("*geometry_param norm_to_plane_thick" + gear_thickness) # Set the thickness of the geometry
    py_send("*geometry_option assumedstrn:on") # Assumed strain active
    py_send("*add_geometry_elements all_existing") #Add this property to all of the geometry
    
    #Element types
    #py_send("*element_type 124 all_existing") #Change all of the elements to plane stress full integration second order
    py_send("*element_type 3 all_existing") # Plane stress full integration quad 1st order
    
    #Flip elements to the appropriate orientation
    py_send("*check_upside_down") # Find and select the upside-down elements
    py_send("*flip_elements all_selected") # Flip the elements that were upside-down
    return

def job_setup():
    py_send("*new_job structural") # Start a new job
    py_send("*add_job_loadcases lcase1") # Use loadcase 1
    py_send("*job_option strain:large") # Use large strain formulation
    py_send("*job_option dimen:pstress") # Plane stress
    
    
    #Job results
    py_send("*add_post_var von_mises") # Add equivalent von mises stress
    
    #Run the Job
    py_send("*update_job")
    py_send("*save_model")
    #py_send("*submit_job 1 *monitor_job")
    #print("sleep_start")
    #time.sleep(7)
    #print("sleep_end")
    
    #py_send("*update_job")
    #py_send("*post_open_default") #Open the default post file
    #py_send("*post_monitor")
    return

def post():
    py_send("*post_contour_bands")

def tables():
    """Creates tables from generated text files"""
    py_send('*md_table_read_any "motion_3.txt"') #Imports the motion table
    py_send('*set_md_table_type 1 time') # X axis of table is time
    
    py_send('*md_table_read_any "moment_3.txt"') #Imports the moment table
    py_send('*set_md_table_type 1 time') # X axis of table is time
    return()

def main():
    py_send("*new_model yes") # Start a new model without saving
    
    py_send("*select_clear")
    
    tables()
    
    #ring_mesh = "Ring_gear_geometry_centred.DAT"
    #ring_mesh = "Ring_gear_geometry_centred.bdf"
    #ring_mesh = "Ring_0305.DAT"
    #ring_mesh = "Ring_0305.UNV"
    #ring_mesh = "healthy_tooth_mesh.bdf"
    ring_bdf = "Ring_Marc_Mesh_0305.bdf"
    import_ring(ring_bdf)
    
    #planet_bdf = "Planet_centred.DAT"
    planet_bdf = "Planet_Marc_Mesh_0305.bdf"
    import_planet(planet_bdf,-1.5,-1.7)
    
    #py_send("*fill_view")
    #py_send("*zoom_box(1,0.451505,0.074545,0.552676,0.180000)") # look at teeth
    
    contact()
    
    loadcase()
    
    geometrical_properties_and_element_types()
    
    material_properties()
    
    job_setup()
    
    post()
    return 

if __name__ == '__main__' :
    py_connect("", 40007)
    main()
    py_disconnect()
