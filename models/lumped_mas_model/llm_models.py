import src.models.lumped_mas_model as pglmm
import numpy as np

def make_chaari_2006_model():
    # Chaari
    # # Number of planet gears
    # ########################################################################################################################
    N = 4

    #  Mass and inertia
    ########################################################################################################################
    m_c = 3  # [kg] Carrier mass
    m_r = 0.588  # [kg] Ring mass
    m_s = 0.46  # [kg] Sun mass
    m_1 = 0.177  # [kg] Planet 1 mass
    m_2, m_3, m_4 = m_1, m_1, m_1  # All planet gears have equal mass

    Ir2_c = 1.5  # [kg] Carrier       #These values are I/r^2
    Ir2_r = 0.759  # [kg] Ring
    Ir2_s = 0.272  # [kg] Sun
    Ir2_1 = 0.1  # [kg] Planet 1
    Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2

    M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3, m_4],
                         [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3, Ir2_4]])

    #  Geometric properties
    ########################################################################################################################
    alpha_s = np.deg2rad(21.34)  # Pressure angle at sun gear
    alpha_r = np.deg2rad(21.34)  # Pressure angle at ring gear

    Geom_atr_ud = np.array([alpha_s, alpha_r])


    #  Stiffness properties
    ########################################################################################################################
    k_Sp = 2* 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness, Peak value?
    k_rp = 2 * 10 ** 8  # [N/m]  # Ring Planet   gear mesh stiffness

    k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)

    k_ru = 10**9 # [N/m]  # Rotational stiffness preventing ring gear from rotating
    k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])

    #  Operating conditions
    ########################################################################################################################
    Omega_c = 0  # 2*np.pi*1285/60#(2*np.pi*8570/60)/(1 + 70/30)#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
    T_s = 100  # [N/m]  # Sun torque applied to the sun gear

    Opp_atr_ud = np.array([Omega_c, T_s])

    PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)
    return PG

def make_lin_1999_model():
    # PARKER 1999
    # Number of planet gears
    #######################################################################################################################
    N = 4
    # #  Mass and inertia
    ########################################################################################################################
    m_c = 5.43  # [kg] Carrier mass
    m_r = 2.35  # [kg] Ring mass
    m_s = 0.4  # [kg] Sun mass
    m_1 = 0.66  # [kg] Planet 1 mass
    m_2, m_3, m_4 = m_1, m_1, m_1  # All planet gears have equal mass
    Ir2_c = 6.29  # [kg] Carrier       #These values are I/r^2
    Ir2_r = 3.0  # [kg] Ring
    Ir2_s = 0.39  # [kg] Sun
    Ir2_1 = 0.61  # [kg] Planet 1
    Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2

    M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3, m_4],
                         [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3, Ir2_4]])
    # #  Geometric properties
    ########################################################################################################################
    alpha_s = np.deg2rad(24.6)  # Pressure angle at sun gear
    alpha_r = np.deg2rad(24.6)  # Pressure angle at ring gear

    Geom_atr_ud = np.array([alpha_s, alpha_r])

    #  Stiffness properties
    ########################################################################################################################
    k_Sp = 5 * 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness
    k_rp = 5 * 10 ** 8  # [N/m]  # Ring-planet   gear mesh stiffness
    k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)

    k_ru = 10**9 # [N/m]  # Rotational stiffness preventing ring gear from rotating
    k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])

    # #  Operating conditions
    # ########################################################################################################################
    Omega_c = 0  # 100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
    T_s = 0  # [N/m]  # Sun torque applied to the sun gear

    Opp_atr_ud = np.array([Omega_c, T_s])

    PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)
    return PG

def make_bonfiglioli_model():
    # Planetary gearbox test bench
    # Number of planet gears
    #######################################################################################################################
    N = 3
    #  Mass and inertia
    ########################################################################################################################
    m_c = 1.11  # [kg] Carrier mass
    m_r = 0.85  # [kg] Ring mass
    m_s = 0.078  # [kg] Sun mass
    m_1 = 00.1938  # [kg] Planet 1 mass
    m_2, m_3 = m_1, m_1  # All planet gears have equal mass
    Ir2_c = 0.556  # [kg] Carrier       #These values are I/r^2
    Ir2_r = 0.7144  # [kg] Ring
    Ir2_s = 0.0393  # [kg] Sun
    Ir2_1 = 0.1237  # [kg] Planet 1
    Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2
    M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3],
                         [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3]])
    #  Geometric properties
    ########################################################################################################################
    alpha_s = np.deg2rad(20)  # Pressure angle at sun gear
    alpha_r = np.deg2rad(20)  # Pressure angle at ring gear

    Geom_atr_ud = np.array([alpha_s, alpha_r])
    # Stiffness properties
    #######################################################################################################################
    k_Sp = 0.1 * 10 ** 7  # [N/m]  # Sun-planet gear mesh stiffness
    k_rp = 0.1 * 10 ** 7  # [N/m]  # Sun-ring   gear mesh stiffness

    k_p = 0.3 * 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)

    k_ru = 10**9 # [N/m]  # Rotational stiffness preventing ring gear from rotating
    k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])

    #  Operating conditions
    ########################################################################################################################
    Omega_c = 0  # 100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
    T_s = 0  # [N/m]  # Sun torque applied to the sun gear

    Opp_atr_ud = np.array([Omega_c, T_s])

    PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)

    return PG

def make_chaari_2006_model_experiment():
    # Chaari
    # # Number of planet gears
    # ########################################################################################################################
    N = 4

    #  Mass and inertia
    ########################################################################################################################
    m_c = 3  # [kg] Carrier mass
    m_r = 0.588  # [kg] Ring mass
    m_s = 0.46  # [kg] Sun mass
    m_1 = 0.177  # [kg] Planet 1 mass
    m_2, m_3, m_4 = m_1, m_1, m_1  # All planet gears have equal mass

    Ir2_c = 1.5  # [kg] Carrier       #These values are I/r^2
    Ir2_r = 0.759  # [kg] Ring
    Ir2_s = 0.272  # [kg] Sun
    Ir2_1 = 0.1  # [kg] Planet 1
    Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2

    M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3, m_4],
                         [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3, Ir2_4]])

    #  Geometric properties
    ########################################################################################################################
    alpha_s = np.deg2rad(21.34)  # Pressure angle at sun gear
    alpha_r = np.deg2rad(21.34)  # Pressure angle at ring gear

    Geom_atr_ud = np.array([alpha_s, alpha_r])


    #  Stiffness properties
    ########################################################################################################################
    k_Sp = 2* 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness, Peak value?
    k_rp = 2 * 10 ** 8  # [N/m]  # Ring Planet   gear mesh stiffness

    k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)

    k_ru = 10**9 # [N/m]  # Rotational stiffness preventing ring gear from rotating
    k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])

    #  Operating conditions
    ########################################################################################################################
    Omega_c = 0  # 2*np.pi*1285/60#(2*np.pi*8570/60)/(1 + 70/30)#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
    T_s = 0  # [N/m]  # Sun torque applied to the sun gear

    Opp_atr_ud = np.array([Omega_c, T_s])

    PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)
    return PG

def make_chaari_2006_1planet():
    # Chaari
    # # Number of planet gears
    # ########################################################################################################################
    N = 1

    #  Mass and inertia
    ########################################################################################################################
    m_c = 3  # [kg] Carrier mass
    m_r = 0.588  # [kg] Ring mass
    m_s = 0.46  # [kg] Sun mass
    m_1 = 0.177  # [kg] Planet 1 mass


    Ir2_c = 1.5  # [kg] Carrier       #These values are I/r^2
    Ir2_r = 0.759  # [kg] Ring
    Ir2_s = 0.272  # [kg] Sun
    Ir2_1 = 0.1  # [kg] Planet 1

    M_atr_ud = np.array([[m_c, m_r, m_s, m_1],
                         [Ir2_c, Ir2_r, Ir2_s, Ir2_1]])

    #  Geometric properties
    ########################################################################################################################
    alpha_s = np.deg2rad(21.34)  # Pressure angle at sun gear
    alpha_r = np.deg2rad(21.34)  # Pressure angle at ring gear

    Geom_atr_ud = np.array([alpha_s, alpha_r])


    #  Stiffness properties
    ########################################################################################################################
    k_Sp = 2* 10 ** 8  # [N/m]  # Sun-planet gear mesh stiffness, Peak value?
    k_rp = 2 * 10 ** 8  # [N/m]  # Ring Planet   gear mesh stiffness

    k_p = 10 ** 8  # [N/m]  # Bearing stiffness (Planet bearing?)

    k_ru = 10**9 # [N/m]  # Rotational stiffness preventing ring gear from rotating
    k_atr_ud = np.array([k_Sp, k_rp, k_p, k_ru])

    #  Operating conditions
    ########################################################################################################################
    Omega_c = (2*np.pi*8570/60)/(1 + 70/30)#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier 2*np.pi*1285/60#
    T_s = 100  # [N/m]  # Sun torque applied to the sun gear

    Opp_atr_ud = np.array([Omega_c, T_s])

    PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)
    return PG

X0 = np.array([-1.67427896e-15,  1.81686928e-07,  1.81686884e-08,  3.54914606e-08,
       -9.08434456e-08, -9.75303772e-09, -3.54914583e-08, -9.08434387e-08,
        9.88800572e-07, -3.34859651e-15,  3.81542498e-07, -5.11431563e-07,
        4.10034386e-14,  9.21551453e-09,  9.21668277e-10,  1.80039978e-09,
       -4.60827729e-09, -4.94760191e-10, -1.80045025e-09, -4.60840343e-09,
        4.77182740e-08,  8.20079192e-14,  1.93539270e-08, -2.59430020e-08])