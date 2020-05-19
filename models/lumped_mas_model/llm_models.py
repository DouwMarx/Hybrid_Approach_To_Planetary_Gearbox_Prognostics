#import src.models.lumped_mas_model as pglmm
import numpy as np


# def make_chaari_2006_model():
#     # Chaari
#     # # Number of planet gears
#     # ########################################################################################################################
#     N = 4
#
#     #  Mass and inertia
#     ########################################################################################################################
#     m_c = 3  # [kg] Carrier mass
#     m_r = 0.588  # [kg] Ring mass
#     m_s = 0.46  # [kg] Sun mass
#     m_1 = 0.177  # [kg] Planet 1 mass
#     m_2, m_3, m_4 = m_1, m_1, m_1  # All planet gears have equal mass
#
#     Ir2_c = 1.5  # [kg] Carrier       #These values are I/r^2
#     Ir2_r = 0.759  # [kg] Ring
#     Ir2_s = 0.272  # [kg] Sun
#     Ir2_1 = 0.1  # [kg] Planet 1
#     Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2
#
#     M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3, m_4],
#                          [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3, Ir2_4]])
#
#     #  Geometric properties
#     ########################################################################################################################
#     alpha_s = np.deg2rad(21.34)  # Pressure angle at sun gear
#     alpha_r = np.deg2rad(21.34)  # Pressure angle at ring gear
#
#     Geom_atr_ud = np.array([alpha_s, alpha_r])
#
#     #  Stiffness properties
#     ########################################################################################################################
#     k_Sp = 2 * 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness, Peak value?
#     k_rp = 2 * 10 ** 8  # [N/m]  # Ring Planet   gear mesh stiffness
#
#     k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)
#
#     k_ru = 10 ** 9  # [N/m]  # Rotational stiffness preventing ring gear from rotating
#     k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])
#
#     #  Operating conditions
#     ########################################################################################################################
#     Omega_c = 2 * np.pi * 1285 / 60  # (2*np.pi*8570/60)/(1 + 70/30)#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
#     T_s = 10  # [N/m]  # Sun torque applied to the sun gear
#
#     Opp_atr_ud = {"Omega_c": Omega_c,
#                   "T_s": T_s,
#                   "base_excitation": False}
#
#     #  Solver attributes
#     ######################################################################################################################
#     timerange = np.linspace(0, 0.051, 10000)
#
#     # X0 = np.array([[-1.67427896e-15, 1.81686928e-07, 1.81686884e-08, 3.54914606e-08,
#     #                 -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
#     #                 9.88800572e-07, -3.34859651e-15, 3.81542498e-07, -5.11431563e-07]]).T
#     # Xd0 = np.array([[4.10034386e-14, 9.21551453e-09, 9.21668277e-10, 1.80039978e-09,
#     #                  -4.60827729e-09, -4.94760191e-10, -1.80045025e-09, -4.60840343e-09,
#     #                  4.77182740e-08, 8.20079192e-14, 1.93539270e-08, -2.59430020e-08]]).T
#
#     X0 = np.zeros((N * 3 + 9, 1))
#     Xd0 = np.zeros((N * 3 + 9, 1))
#
#     solve_atr = {"solver_alg": "Radau",
#                  "proportional_damping_constant": 0.03,
#                  "time_varying_proportional_damping": False,
#                  "X0": X0,
#                  "Xd0": Xd0,
#                  "time_range": timerange
#                  }
#
#     PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud, solve_atr)
#     return PG


def make_chaari_2006_model_w_dict():
    # Gearbox layout
    # ##################################################################################################################
    # Number of planet gears
    N = 4

    # Number of gear teeth
    Z_r = 70
    Z_s = 30
    Z_p = 20

    # Pressure angles
    alpha_s = np.deg2rad(21.34)  # Pressure angle at sun gear
    alpha_r = np.deg2rad(21.34)  # Pressure angle at ring gear

    gearbox_layout = {"N": N,
                      "Z_r": Z_r,
                      "Z_s": Z_s,
                      "Z_p": Z_p,
                      "alpha_s": alpha_s,
                      "alpha_r": alpha_r}

    #  Mass and inertia
    ####################################################################################################################
    m_c = 3.0  # [kg] Carrier mass
    m_r = 0.588  # [kg] Ring mass
    m_s = 0.46  # [kg] Sun mass
    m_p = 0.177  # [kg] Planet 1 mass

    Ir2_c = 1.5  # [kg] Carrier       #These values are I/r^2
    Ir2_r = 0.759  # [kg] Ring
    Ir2_s = 0.272  # [kg] Sun
    Ir2_p = 0.1  # [kg] Planet

    m_atr = {"m_c": m_c,
             "m_r": m_r,
             "m_s": m_s,
             "m_p": m_p,
             "Ir2_c": Ir2_c,
             "Ir2_r": Ir2_r,
             "Ir2_s": Ir2_s,
             "Ir2_p": Ir2_p}

    # Stiffness properties
    ####################################################################################################################
    k_Sp = 2 * 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness
    k_rp = 2 * 10 ** 8  # [N/m]  # Ring-Planet gear mesh stiffness
    k_p = 10 ** 8  # [N/m]  # Bearing stiffness
    k_ru = 10 ** 9  # [N/m]  # Rotational stiffness preventing ring gear from rotating

    delta_k = 0.5*k_rp

    k_atr = {"k_Sp": k_Sp,
             "k_rp": k_rp,
             "delta_k": delta_k,
             "k_p": k_p,
             "k_ru": k_ru}

    #  Operating conditions
    ####################################################################################################################
    Omega_c = 2 * np.pi * 1285 / 60  # [rad/s]  # Constant angular speed of planet carrier
    T_s = 10  # [N/m]  # Sun torque applied to the sun gear

    opp_atr = {"Omega_c": Omega_c,
               "T_s": T_s,
               "base_excitation": False}

    #  Solver attributes
    ####################################################################################################################
    timerange = np.linspace(0, 0.051, 10000)

    X0 = np.random.rand(N*3 + 9, 1)*1E-6
    Xd0 = np.random.rand(N*3 + 9, 1)*1E-6

    #X0 = np.zeros((N * 3 + 9, 1))
    #Xd0 = np.zeros((N * 3 + 9, 1))

    solve_atr = {"solver_alg": "Radau",
                 "proportional_damping_constant": 0.003,
                 "time_varying_proportional_damping": False,
                 "X0": X0,
                 "Xd0": Xd0,
                 "time_range": timerange
                 }

    PG_info = {"gearbox_layout": gearbox_layout,
               "m_atr": m_atr,
               "k_atr": k_atr,
               "opp_atr": opp_atr,
               "solve_atr": solve_atr}
    return PG_info


# def make_lin_1999_model():
#     # PARKER 1999
#     # Number of planet gears
#     #######################################################################################################################
#     N = 4
#     # #  Mass and inertia
#     ########################################################################################################################
#     m_c = 5.43  # [kg] Carrier mass
#     m_r = 2.35  # [kg] Ring mass
#     m_s = 0.4  # [kg] Sun mass
#     m_1 = 0.66  # [kg] Planet 1 mass
#     m_2, m_3, m_4 = m_1, m_1, m_1  # All planet gears have equal mass
#     Ir2_c = 6.29  # [kg] Carrier       #These values are I/r^2
#     Ir2_r = 3.0  # [kg] Ring
#     Ir2_s = 0.39  # [kg] Sun
#     Ir2_1 = 0.61  # [kg] Planet 1
#     Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2
#
#     M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3, m_4],
#                          [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3, Ir2_4]])
#     # #  Geometric properties
#     ########################################################################################################################
#     alpha_s = np.deg2rad(24.6)  # Pressure angle at sun gear
#     alpha_r = np.deg2rad(24.6)  # Pressure angle at ring gear
#
#     Geom_atr_ud = np.array([alpha_s, alpha_r])
#
#     #  Stiffness properties
#     ########################################################################################################################
#     k_Sp = 5 * 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness
#     k_rp = 5 * 10 ** 8  # [N/m]  # Ring-planet   gear mesh stiffness
#     k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)
#
#     k_ru = 10 ** 9  # [N/m]  # Rotational stiffness preventing ring gear from rotating
#     k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])
#
#     #  Operating conditions
#     ########################################################################################################################
#     Omega_c = 0  # 2*np.pi*1285/60#(2*np.pi*8570/60)/(1 + 70/30)#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
#     T_s = 100  # [N/m]  # Sun torque applied to the sun gear
#
#     Opp_atr_ud = {"Omega_c": Omega_c,
#                   "T_s": T_s,
#                   "base_excitation": False}
#
#     #  Solver attributes
#     ######################################################################################################################
#     timerange = np.linspace(0, 0.1, 100000)
#
#     # X0 = np.array([[-1.67427896e-15, 1.81686928e-07, 1.81686884e-08, 3.54914606e-08,
#     #                 -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
#     #                 9.88800572e-07, -3.34859651e-15, 3.81542498e-07, -5.11431563e-07]]).T
#     # Xd0 = np.array([[4.10034386e-14, 9.21551453e-09, 9.21668277e-10, 1.80039978e-09,
#     #                  -4.60827729e-09, -4.94760191e-10, -1.80045025e-09, -4.60840343e-09,
#     #                  4.77182740e-08, 8.20079192e-14, 1.93539270e-08, -2.59430020e-08]]).T
#
#     X0 = np.zeros((N * 3 + 9, 1))
#     Xd0 = np.zeros((N * 3 + 9, 1))
#
#     solve_atr = {"solver_alg": "Radau",
#                  "proportional_damping_constant": 0.03,
#                  "time_varying_proportional_damping": False,
#                  "X0": X0,
#                  "Xd0": Xd0,
#                  "time_range": timerange
#                  }
#
#     PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud, solve_atr)
#     return PG
#
#
# def make_bonfiglioli_model():
#     # Planetary gearbox test bench
#     # Number of planet gears
#     #######################################################################################################################
#     N = 3
#     #  Mass and inertia
#     ########################################################################################################################
#     m_c = 1.11  # [kg] Carrier mass
#     m_r = 0.85  # [kg] Ring mass
#     m_s = 0.078  # [kg] Sun mass
#     m_1 = 00.1938  # [kg] Planet 1 mass
#     m_2, m_3 = m_1, m_1  # All planet gears have equal mass
#     Ir2_c = 0.556  # [kg] Carrier       #These values are I/r^2
#     Ir2_r = 0.7144  # [kg] Ring
#     Ir2_s = 0.0393  # [kg] Sun
#     Ir2_1 = 0.1237  # [kg] Planet 1
#     Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2
#     M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3],
#                          [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3]])
#     #  Geometric properties
#     ########################################################################################################################
#     alpha_s = np.deg2rad(20)  # Pressure angle at sun gear
#     alpha_r = np.deg2rad(20)  # Pressure angle at ring gear
#
#     Geom_atr_ud = np.array([alpha_s, alpha_r])
#     # Stiffness properties
#     #######################################################################################################################
#     k_Sp = 0.1 * 10 ** 7  # [N/m]  # Sun-planet gear mesh stiffness
#     k_rp = 0.1 * 10 ** 7  # [N/m]  # Sun-ring   gear mesh stiffness
#
#     k_p = 0.3 * 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)
#
#     k_ru = 10 ** 9  # [N/m]  # Rotational stiffness preventing ring gear from rotating
#     k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])
#
#     #  Operating conditions
#     ########################################################################################################################
#     Omega_c = 0  # 100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
#     T_s = 0  # [N/m]  # Sun torque applied to the sun gear
#
#     Opp_atr_ud = np.array([Omega_c, T_s])
#
#     PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)
#
#     return PG
#
#
# def make_chaari_2006_model_experiment():
#     # Chaari
#     # # Number of planet gears
#     # ########################################################################################################################
#     N = 4
#
#     #  Mass and inertia
#     ########################################################################################################################
#     m_c = 3  # [kg] Carrier mass
#     m_r = 0.588  # [kg] Ring mass
#     m_s = 0.46  # [kg] Sun mass
#     m_1 = 0.177  # [kg] Planet 1 mass
#     m_2, m_3, m_4 = m_1, m_1, m_1  # All planet gears have equal mass
#
#     Ir2_c = 1.5  # [kg] Carrier       #These values are I/r^2
#     Ir2_r = 0.759  # [kg] Ring
#     Ir2_s = 0.272  # [kg] Sun
#     Ir2_1 = 0.1  # [kg] Planet 1
#     Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2
#
#     M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3, m_4],
#                          [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3, Ir2_4]])
#
#     #  Geometric properties
#     ########################################################################################################################
#     alpha_s = np.deg2rad(21.34)  # Pressure angle at sun gear
#     alpha_r = np.deg2rad(21.34)  # Pressure angle at ring gear
#
#     Geom_atr_ud = np.array([alpha_s, alpha_r])
#
#     #  Stiffness properties
#     ########################################################################################################################
#     k_Sp = 2 * 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness, Peak value?
#     k_rp = 2 * 10 ** 8  # [N/m]  # Ring Planet   gear mesh stiffness
#
#     k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)
#
#     k_ru = 10 ** 9  # [N/m]  # Rotational stiffness preventing ring gear from rotating
#     k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])
#
#     #  Operating conditions
#     ########################################################################################################################
#     Omega_c = 0  # 2*np.pi*1285/60#(2*np.pi*8570/60)/(1 + 70/30)#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
#     T_s = 0  # [N/m]  # Sun torque applied to the sun gear
#
#     Opp_atr_ud = np.array([Omega_c, T_s])
#
#     PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)
#     return PG
#
#
# def make_bonfiglioli_full_planet():
#     # Number of planet gears
#     ####################################################################################################################
#     N = 1
#
#     #  Mass and inertia
#     ####################################################################################################################
#     m_c = 1.714  # [kg] Carrier mass
#     m_r = 1.115  # [kg] Ring mass
#     m_s = 0.242  # [kg] Sun mass
#     m_1 = 0.153  # [kg] Planet 1 mass
#
#     Ir2_c = 0.214  # [kg] Carrier       #These values are I/r^2
#     Ir2_r = 0.373  # [kg] Ring
#     Ir2_s = 0.030  # [kg] Sun
#     Ir2_1 = 0.025  # [kg] Planet 1
#
#     M_atr_ud = np.array([[m_c, m_r, m_s, m_1],
#                          [Ir2_c, Ir2_r, Ir2_s, Ir2_1]])
#
#     #  Geometric properties
#     ####################################################################################################################
#     alpha_s = np.deg2rad(20)  # Pressure angle at sun gear
#     alpha_r = np.deg2rad(20)  # Pressure angle at ring gear
#
#     Geom_atr_ud = np.array([alpha_s, alpha_r])
#
#     # Stiffness properties
#     ####################################################################################################################
#     k_Sp = 2.32e6  # [N/m]  # Sun-planet gear mesh stiffness   #2.32e6 -> 2.42e6
#     k_rp = 2.32e6  # [N/m]  # Ring Planet   gear mesh stiffness
#
#     k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)
#
#     k_ru = 10 ** 9  # [N/m]  # Rotational stiffness preventing ring gear from rotating
#     k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])
#
#     #  Operating conditions
#     ####################################################################################################################
#     Omega_c = (2 * np.pi * 8570 / 60) / (
#             62 / 13 + 1)  # 100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier 2*np.pi*1285/60#
#     T_s = 10  # [N/m]  # Sun torque applied to the sun gear
#
#     Opp_atr_ud = {"Omega_c": Omega_c,
#                   "T_s": T_s,
#                   "base_excitation": False}
#
#     #  Solver attributes
#     ####################################################################################################################
#     timerange = np.linspace(0, 0.01, 100000)
#
#     # X0 = np.array([[-1.67427896e-15, 1.81686928e-07, 1.81686884e-08, 3.54914606e-08,
#     #                 -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
#     #                 9.88800572e-07, -3.34859651e-15, 3.81542498e-07, -5.11431563e-07]]).T
#     # Xd0 = np.array([[4.10034386e-14, 9.21551453e-09, 9.21668277e-10, 1.80039978e-09,
#     #                  -4.60827729e-09, -4.94760191e-10, -1.80045025e-09, -4.60840343e-09,
#     #                  4.77182740e-08, 8.20079192e-14, 1.93539270e-08, -2.59430020e-08]]).T
#
#     X0 = np.zeros((12, 1))
#     Xd0 = np.zeros((12, 1))
#
#     solve_atr = {"solver_alg": "Radau",
#                  "proportional_damping_constant": 0.03,
#                  "time_varying_proportional_damping": False,
#                  "X0": X0,
#                  "Xd0": Xd0,
#                  "time_range": timerange
#                  }
#
#     PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud, solve_atr)
#     return PG
#
#
# def make_chaari_2006_1planet():
#     # Number of planet gears
#     ####################################################################################################################
#     N = 1
#
#     #  Mass and inertia
#     ####################################################################################################################
#     m_c = 3  # [kg] Carrier mass
#     m_r = 0.588  # [kg] Ring mass
#     m_s = 0.46  # [kg] Sun mass
#     m_1 = 0.177  # [kg] Planet 1 mass
#
#     Ir2_c = 1.5  # [kg] Carrier       #These values are I/r^2
#     Ir2_r = 0.759  # [kg] Ring
#     Ir2_s = 0.272  # [kg] Sun
#     Ir2_1 = 0.1  # [kg] Planet 1
#
#     M_atr_ud = np.array([[m_c, m_r, m_s, m_1],
#                          [Ir2_c, Ir2_r, Ir2_s, Ir2_1]])
#
#     #  Geometric properties
#     ####################################################################################################################
#     alpha_s = np.deg2rad(21.34)  # Pressure angle at sun gear
#     alpha_r = np.deg2rad(21.34)  # Pressure angle at ring gear
#
#     Geom_atr_ud = np.array([alpha_s, alpha_r])
#
#     # Stiffness properties
#     ####################################################################################################################
#     k_Sp = 2 * 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness, Peak value?
#     k_rp = 2 * 10 ** 8  # [N/m]  # Ring Planet   gear mesh stiffness
#
#     k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)
#
#     k_ru = 10 ** 9  # [N/m]  # Rotational stiffness preventing ring gear from rotating
#     k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])
#
#     #  Operating conditions
#     ####################################################################################################################
#     Omega_c = (2 * np.pi * 8570 / 60) / (
#             1 + 70 / 30)  # 100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier 2*np.pi*1285/60#
#     T_s = 10  # [N/m]  # Sun torque applied to the sun gear
#
#     Opp_atr_ud = {"Omega_c": Omega_c,
#                   "T_s": T_s,
#                   "base_excitation": False}
#
#     #  Solver attributes
#     ####################################################################################################################
#     timerange = np.linspace(0, 0.1, 100000)
#
#     # X0 = np.array([[-1.67427896e-15, 1.81686928e-07, 1.81686884e-08, 3.54914606e-08,
#     #                 -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
#     #                 9.88800572e-07, -3.34859651e-15, 3.81542498e-07, -5.11431563e-07]]).T
#     # Xd0 = np.array([[4.10034386e-14, 9.21551453e-09, 9.21668277e-10, 1.80039978e-09,
#     #                  -4.60827729e-09, -4.94760191e-10, -1.80045025e-09, -4.60840343e-09,
#     #                  4.77182740e-08, 8.20079192e-14, 1.93539270e-08, -2.59430020e-08]]).T
#
#     X0 = np.zeros((12, 1))
#     Xd0 = np.zeros((12, 1))
#
#     solve_atr = {"solver_alg": "Radau",
#                  "proportional_damping_constant": 0.03,
#                  "time_varying_proportional_damping": False,
#                  "X0": X0,
#                  "Xd0": Xd0,
#                  "time_range": timerange
#                  }
#
#     PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud, solve_atr)
#     return PG
#
#
# def make_liang_2015():
#     # Number of planet gears
#     ####################################################################################################################
#     N = 4
#
#     #  Mass and inertia
#     ####################################################################################################################
#     m_c = 10  # [kg] Carrier mass
#     m_r = 5.982  # [kg] Ring mass
#     m_s = 0.700  # [kg] Sun mass
#     m_1 = 1.822  # [kg] Planet 1 mass
#     m_2, m_3, m_4 = m_1, m_1, m_1
#
#     Ir2_c = 5  # [kg] Carrier       #These values are I/r^2    Iz = 0.5*MR^2 for disk
#     Ir2_r = 0.759  # [kg] Ring  # Not mentioned and cannot be calculated without ring gear specs?
#     Ir2_s = 0.35  # [kg] Sun
#     Ir2_1 = 0.911  # [kg] Planet 1
#     Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1
#
#     M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3, m_4],
#                          [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3, Ir2_4]])
#
#     #  Geometric properties
#     ####################################################################################################################
#     alpha_s = np.deg2rad(20)  # Pressure angle at sun gear
#     alpha_r = np.deg2rad(20)  # Pressure angle at ring gear
#
#     Geom_atr_ud = np.array([alpha_s, alpha_r])
#
#     # Stiffness properties
#     ####################################################################################################################
#     k_Sp = 1.2e9  # [N/m]  # Sun-planet gear mesh stiffness, Peak value?
#     k_rp = 1.2e9  # [N/m]  # Ring Planet   gear mesh stiffness
#
#     k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)
#
#     k_ru = 10 ** 9  # [N/m]  # Rotational stiffness preventing ring gear from rotating
#     k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])
#
#     #  Operating conditions
#     ####################################################################################################################
#     Omega_c = 2 * np.pi * 950 / 60  # [rad/s]  # Constant angular speed of planet carrier
#     T_s = 450  # [N/m]  # Sun torque applied to the sun gear
#
#     Opp_atr_ud = {"Omega_c": Omega_c,
#                   "T_s": T_s,
#                   "base_excitation": False}
#
#     #  Solver attributes
#     ######################################################################################################################
#     timerange = np.linspace(0, 0.1, 100000)
#
#     # X0 = np.array([[-1.67427896e-15, 1.81686928e-07, 1.81686884e-08, 3.54914606e-08,
#     #                 -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
#     #                 9.88800572e-07, -3.34859651e-15, 3.81542498e-07, -5.11431563e-07]]).T
#     # Xd0 = np.array([[4.10034386e-14, 9.21551453e-09, 9.21668277e-10, 1.80039978e-09,
#     #                  -4.60827729e-09, -4.94760191e-10, -1.80045025e-09, -4.60840343e-09,
#     #                  4.77182740e-08, 8.20079192e-14, 1.93539270e-08, -2.59430020e-08]]).T
#
#     X0 = np.zeros((N * 3 + 9, 1))
#     Xd0 = np.zeros((N * 3 + 9, 1))
#
#     solve_atr = {"solver_alg": "Radau",
#                  "proportional_damping_constant": 0.03,
#                  "time_varying_proportional_damping": False,
#                  "X0": X0,
#                  "Xd0": Xd0,
#                  "time_range": timerange
#                  }
#
#     PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud, solve_atr)
#     return PG
