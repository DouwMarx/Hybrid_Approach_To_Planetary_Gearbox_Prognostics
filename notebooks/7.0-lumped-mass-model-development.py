import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
#import importlib  #This allows me to reload my own module every time
#importlib.reload(pglmm)
#from pglmm import *
#from mpl_toolkits.mplot3d import Axes3D

plt.close("all")

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
k_Sp = 2*10**8  # [N/m]  # Sun-planet gear mesh stiffness, Peak value?
k_rp = 2*10**8  # [N/m]  # Sun-ring   gear mesh stiffness

k_p = 10**8     # [N/m]  # Bearing stiffness (Planet bearing?)

k_atr_ud = np.array([k_Sp, k_rp, k_p])


#  Operating conditions
########################################################################################################################
Omega_c = (2*np.pi*8570/60)/(1 + 70/30)#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
T_s = 4 # [N/m]  # Sun torque applied to the sun gear


Opp_atr_ud = np.array([Omega_c, T_s])




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# #PARKER 1999
# #Number of planet gears
# #######################################################################################################################
# N = 4
#
# #  Mass and inertia
# ########################################################################################################################
# m_c = 5.43  # [kg] Carrier mass
# m_r = 2.35  # [kg] Ring mass
# m_s = 0.4  # [kg] Sun mass
# m_1 = 0.66  # [kg] Planet 1 mass
# m_2, m_3, m_4 = m_1, m_1, m_1  # All planet gears have equal mass
#
# Ir2_c = 6.29  # [kg] Carrier       #These values are I/r^2
# Ir2_r = 3.0  # [kg] Ring
# Ir2_s = 0.39  # [kg] Sun
# Ir2_1 = 0.61# [kg] Planet 1
# Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2
#
# M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3, m_4],
#                   [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3, Ir2_4]])
#
# #  Geometric properties
# ########################################################################################################################
# alpha_s = np.deg2rad(24.6)  # Pressure angle at sun gear
# alpha_r = np.deg2rad(24.6)  # Pressure angle at ring gear
#
# Geom_atr_ud = np.array([alpha_s, alpha_r])
#
#
# #  Stiffness properties
# ########################################################################################################################
# k_Sp = 5*10**8  # [N/m]  # Sun-planet gear mesh stiffness
# k_rp = 5*10**8  # [N/m]  # Sun-ring   gear mesh stiffness
#
# k_p = 10**8     # [N/m]  # Bearing stiffness (Planet bearing?)
#
# k_atr_ud = np.array([k_Sp, k_rp, k_p])
#
#
# #  Operating conditions
# ########################################################################################################################
# Omega_c = 0#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
# T_s = 0  # [N/m]  # Sun torque applied to the sun gear
#
#
# Opp_atr_ud = np.array([Omega_c, T_s])



# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Planetary gearbox test bench
# #Number of planet gears
# #######################################################################################################################
# N = 3
#
# #  Mass and inertia
# ########################################################################################################################
# m_c = 1.11 # [kg] Carrier mass
# m_r = 0.85  # [kg] Ring mass
# m_s = 0.078  # [kg] Sun mass
# m_1 = 00.1938 # [kg] Planet 1 mass
# m_2, m_3 = m_1, m_1 # All planet gears have equal mass
#
# Ir2_c = 0.556  # [kg] Carrier       #These values are I/r^2
# Ir2_r = 0.7144 # [kg] Ring
# Ir2_s = 0.0393  # [kg] Sun
# Ir2_1 = 0.1237# [kg] Planet 1
# Ir2_2, Ir2_3, Ir2_4 = Ir2_1, Ir2_1, Ir2_1  # All planet gears have equal I/r^2
#
# M_atr_ud = np.array([[m_c, m_r, m_s, m_1, m_2, m_3],
#                   [Ir2_c, Ir2_r, Ir2_s, Ir2_1, Ir2_2, Ir2_3]])
#
# #  Geometric properties
# ########################################################################################################################
# alpha_s = np.deg2rad(20)  # Pressure angle at sun gear
# alpha_r = np.deg2rad(20)  # Pressure angle at ring gear
#
# Geom_atr_ud = np.array([alpha_s, alpha_r])
#
#
# #  Stiffness properties
# ########################################################################################################################
# k_Sp = 0.1*10**7  # [N/m]  # Sun-planet gear mesh stiffness
# k_rp = 0.1*10**7  # [N/m]  # Sun-ring   gear mesh stiffness
#
# k_p = 0.3*10**8     # [N/m]  # Bearing stiffness (Planet bearing?)
#
# k_atr_ud = np.array([k_Sp, k_rp, k_p])
#
#
# #  Operating conditions
# ########################################################################################################################
# Omega_c = 0#100*2*np.pi*60  # [rad/s]  # Constant angular speed of planet carrier
# T_s = 0  # [N/m]  # Sun torque applied to the sun gear
#
#
# Opp_atr_ud = np.array([Omega_c, T_s])





#######################################################################################################################


PG = pglmm.Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)
t = np.linspace(0,0.1,100)

d1 = pglmm.DE_Integration(PG)
X0 = d1.X_0()

# #X0[4] = 0.00001
# #X0[0] = 10
#
sol = d1.Run_Integration(X0, t)

#
plt.figure()
plt.plot(sol)


def try_match_chaari():
    #t = 0

    # t = np.array([0.8, 0.8])
    #
    # def obj(t):
    #     K = PG.K_b + PG.K_e(t)
    #
    #     val, vec = sci.linalg.eig(K, PG.M)
    #
    #     summation =0
    #     for f in [0, 1998,3166,3938,7998,8889,10490]:
    #         summation += np.min((np.real(np.sqrt(val) / (np.pi * 2)) - f)**2)/(1+f)**2
    #
    #     return summation

    import scipy.optimize as opt

    #sol = opt.minimize(obj,t,bounds=((0.5,1),(0.5,1)),tol=10e-10)
    #print(sol)
    #t = sol["x"]


    #

    #
    # for i in range(len(val)):
    #     print([np.sqrt(val[i])/(np.pi*2),vec[i][0:3]])
    #
    # # Muu = PG.M[3:, 3:]
    # # Kuu = K[3:, 3:]
    # # #
    # # #
    # # val, vec = sci.linalg.eig(Kuu, Muu)
    # # #
    # # print(np.sort(np.sqrt(val)/(np.pi*2)))


    # t1 = np.linspace(0.5,0.9,100)
    # t2 = np.linspace(0.5,0.9,100)
    #
    # T1,T2 = np.meshgrid(t1,t2)

    # fvals = []
    # for t_1 in t1:
    #     for t_2 in t2:
    #         fvals.append(obj([t_1, t_2]))
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # surf = ax.plot_surface(T1,T2,np.array(fvals).reshape(len(t1),len(t1)))
    # plt.show()
    return

def plots():
    #Making plots for MSS report
    PG = Planetary_Gear(N, M_atr_ud, Geom_atr_ud, k_atr_ud, Opp_atr_ud)  # Create the planetary gear object
    ke1 = K_e(PG)
    kb1 = K_b(PG)

    t = 1
    K = PG.K_b + PG.K_e(t)
    val, vec = sci.linalg.eig(K, PG.M)
    print(np.sort(np.sqrt(val) / (np.pi * 2)))

    traange = np.linspace(0,0.02,10000)
    stiffness = ke1.k_sp(traange)

    plt.figure()
    plt.plot(traange, stiffness, "k")
    plt.xlabel("Time [s]")
    plt.ylabel("Gear Mesh Stiffness [N/m]")
    plt.savefig("TVMS.pdf")

    import matplotlib


    count = 1
    for t in [0, 0.0025]:
        plt.figure()
        Kmat = PG.K_e(t) + PG.K_b + PG.K_Omega
        plt.imshow(Kmat, vmin=-400000000.0, vmax=1600000000.0, cmap="gist_heat")
        cbar = plt.colorbar()#norm=matplotlib.colors.LogNorm())
        cbar.ax.set_ylabel("Stiffness [N/m]")
        plt.axis('off')
        #name = "Mesh_stiff" + str(count) + ".pdf"

        name = "Total_stiff" + str(count) + ".pdf"
        plt.savefig(name)

        count+=1
        #print(np.min(Kmat))
        #print(np.max(Kmat))
    return