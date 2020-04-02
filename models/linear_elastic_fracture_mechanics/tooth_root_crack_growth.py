import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as inter
plt.close("all")

# Fatigue Calculation based on
# - https://mechanicalc.com/reference/stress-intensity-factor-solutions
# - http://www.faculty.fairfield.edu/wdornfeld/ME312/ToothLoads04.pdf
# - Dowling - Mechanical behaviour of materials


#  Assumptions
# 1) For calculating the Lewis bending stresses,
#    the additional stresses due to high velocity gear operation is not included
#    this is considered conservative if we want the crack to growth
# 2) Stress intensity is based on Single Edge Through crack in a plate.
#    No contribution due to axial load; bending only.
# 3) Assume the fracture toughness of the material is 20Mpa*m^0.5 - The upper bound of fracture toughness of cast irons
# 4) Assume zero to tensile loading, R=0.
# 5) Assume the motor can deliver 90% of its actual power rating of 3kW
# 6) Assume the force acting on the gear tooth is the force for single teeth engagement
# 7) Assume 10% torque losses due to long gear train
# 8) Assume the cracks will be grown to 80% of the number of cycles to reach critical Crack length (End of life).
#    More difficult to predict crack length as the crack starts growing faster


class Opperating_Conditions(object):
    def __init__(self, phi, Zr, Zs, Zp, m, Motor_speed):
        """Give motor speed in RPM"""

        self.phi = phi # The pressure angle of the planetary gearbox, specify in degrees
        self.Zs = Zs  # Number of teeth of the sun gear
        self.Zp = Zp  # Number of teeth of the planet gear
        self.Zr = Zr  # Number of teeth of ring gear
        self.m = m  # Gear module
        self.speed = 2*np.pi*Motor_speed/60  # rad/s
        self.GR = self.GR_calc() # Calculate the gear ratio of the gearbox
        self.T = self.Torque(self.speed) # Calculates the torque produced by the motor
        self.Planet_tooth_force = self.F_PS() # Calculates the force on the planet gear tooth
        self.f_fault = self.F_planet_fault() # calculate the fault frequency of the planet gear with damaged tooth

    def Torque(self,rspeed):
        ''' Motor speed in RPM, Using a 3KW motor
        https://www.motioncontroltips.com/torque-equation/'''
        #V = 220  # Motor supply voltage
        #k = 1 # Motor constant
        #R = 1000  # Motor resistance

        #T = k*(V-Motor_speed*k)/R

        #  9.5 is the maximum torque @ 3000RPM considering that the motor is rated for 3kW @ 3000RPM
        return 40 - rspeed*(40-9)/(3000)

    def F_PS(self):
        '''Calculates the force on the planet gear tooth from the sun gear if it was under single tooth contact
        https://engineering.stackexchange.com/questions/16136/calculating-load-on-planetary-gear-from-driving-torque'''
        Ti = self.T
        n_eff = 0.9
        Ti = Ti*n_eff  # Assume some losses take place in the gear train
        #F = Ti/(3*0.5*self.Zs*self.m*np.cos(np.deg2rad(self.phi)))  # For three planet carriers
        F = Ti / (1* 0.5 * self.Zs * self.m * np.cos(np.deg2rad(self.phi)))  # Notice this is for one planet gear
        return F


    def GR_calc(self):
        """Gear ratio for planetary gearbox with stationary ring gear
        Parameters
        ----------


        Returns
        -------
        GR: Float
            The ratio
            """
        return 1 + self.Zr / self.Zs


    def f_p(self,f_c):
        """Frequency of rotation of planetary gears
        Parameters
        ----------
        f_c: Float
             Frequency of rotation of planet carrier


        Returns
        -------
        f_p: Float
            Frequency of rotation of planet gear
            """
        return f_c*self.Zr/self.Zp

    def GMF(self):
        """Calculates the gear mesh frequency for a given a sun gear input frequency. The gearbox is therefore running in the speed down configuration
        Parameters
        ----------
        f_s: Float
             Frequency of rotation of sun gear


        Returns
        -------
        GMF: Float
            Frequency of rotation of planet gear
            """
        f_s = self.speed/(2*np.pi)  # Shaft frequency [Hz]
        fc = f_s/self.GR
        return self.f_p(fc)*self.Zp

    def F_planet_fault(self):
        "The fault frequency of the planet gear"
        F_fault = self.GMF()/self.Zp #  Divide by the number of planet teeth for the fault frequency
        return F_fault


class Cracked_Gear_Tooth(object):
    def __init__(self,a, b, F, Pd, K1c, C, M, Y, Opper):

        self.ai = a # Initial crack length
        self.b = b  # Tooth breadth at the root, used in edge through crack calcs
        self.F = F  # Gear Face width
        self.Pd = Pd  # Diametral pitch

        self.K1c = K1c # Plane stain fracture toughness
        self.C = C # Paris law constant 1
        self.M = M # Paris law constant 1

        self.Y = Y  # Lewis form factor

        self.Opper = Opper  # Operating condition object

        self.sigma_bnd = self.Lewis_Bending_Stress() # Calculate the lewis bending stress
        self.a_c = self.Critical_Crack_Length()  # Calculate the critical crack length

        self.a_list, self.N_list = self.Paris_integration(self.ai, self.a_c)
        self.EOL = 0.8*self.N_list[-1]  #  End of life defined as 80% of the critical crack length

        self.t_EOL = self.Time_to_EOL()


    def Lewis_Bending_Stress(self):
        ''' This calculates the lewis bending stresses (Without increased stresses due to Barth Velocity Factor'''
        Wt = np.cos(self.Opper.phi)*self.Opper.Planet_tooth_force #  Tangential load
        return Wt*self.Pd/(self.F*self.Y)

    def Geometry_factor_bending(self, a):
        # See figure 8.13 in Dowling - Making use of ASTM standard for h/b = 2 bend specimen
        alpha_sec = a/self.b
        #sqrt_arg = 2*np.tan(np.pi*alpha_sec/2)/(np.pi*alpha_sec)
        #top = 0.923 + 0.199*(1 - np.sin(np.pi*alpha_sec/2))**4
        #bot = np.cos(np.pi*alpha_sec/2)
        #Yb = np.sqrt(sqrt_arg)*top/bot

        top = 1.99 - alpha_sec*(1-alpha_sec)*(2.15 - 3.93*alpha_sec + 2.7*alpha_sec**2)
        bot = np.sqrt(np.pi)*(1+2*alpha_sec)*(1-alpha_sec)**(3/2)
        Yb = top/bot
        return Yb

    def Stess_Intensity(self, a):
        Yb = self.Geometry_factor_bending(a)
        K = Yb*self.sigma_bnd*np.sqrt(np.pi*a)  # sigma_t is sigma_b, the lewis bending stress
        return K

    def Critical_Crack_Length(self):
        obj = lambda a: self.Stess_Intensity(a) - self.K1c
        sol = opt.fsolve(obj, 0.9*self.b) #Solve for the critical crack length, 0.9b is just a good starting point
        return sol[0]

    def dadN(self, a, N):
        return self.C*1e-3*(self.Stess_Intensity(a)*1e-6)**self.M  # Notice *1e-3 to work in m/cycle. Also, stress
                                                                   # stress intensity range in MPa sqrt(m)

    def dNda(self, a, N):
        '''Inverse of Paris law'''
        return 1/self.dadN(a,N)

    def Paris_integration(self,ai,af):
        """Returns the Number of cycles for a given crack length"""

        # I make use of Euler integration because it is simple. Aslo scipy.integrate.odeint give problems.

        increms = int(1e3)  # This gives a good trade off between accuracy and speed
        a_range = np.logspace(np.log10(ai), np.log10(af), increms)#  Using a non-linear time scale helps with accuracy

        N = 1
        #
        Nlist = [N]
        for i in range(len(a_range) - 1):
            delta_a = a_range[i + 1] - a_range[i]  # Calculates the length of the integration increment
            a = a_range[i]
            N = N + self.dNda(a, "dummy") * delta_a
            Nlist.append(N)
        return a_range, np.array(Nlist)

    def Time_to_EOL(self):
        "Gives the time to EOL in hours based on the fault frequency"
        t = self.EOL/self.Opper.f_fault  # Time in seconds
        t = t/3600  # Time in hours
        return t


class Plots(object):
    def __init__(self,crack_obj,opper_obj):
        self.crack_obj = crack_obj
        self.opper_obj = opper_obj


    def Plot_Brittle_Fracture(self):
        lengths = np.linspace(0, self.crack_obj.a_c*1.02*1000, 1000)

        plt.figure()
        plt.title("Brittle fracture")
        plt.plot(lengths, self.crack_obj.Stess_Intensity(lengths / 1000) / 1e6)
        plt.plot([lengths[0], lengths[-1]], [self.crack_obj.K1c / 1e6, self.crack_obj.K1c / 1e6])
        plt.xlabel("Crack Length [mm]")
        plt.ylabel("Stress intensity factor MPa m^2")
        text = "Critical crack length = " + str(np.round(self.crack_obj.a_c*1000, 2)) + " mm"
        plt.text(0.1/1000, self.crack_obj.K1c*1.1/1e6, text)

    def Plot_Torque_vs_speed(self):
        speed = np.linspace(0,3000,100)
        torque = self.opper_obj.Torque(speed)

        plt.figure()
        plt.title("Torque that motor is capable of delivering")
        plt.plot(speed, torque)
        plt.xlabel("Speed [RPM]")
        plt.ylabel("Torque [N.m]")
        plt.show()

    def Plot_N_vs_a(self):
        a_list, N_list = self.crack_obj.a_list, self.crack_obj.N_list
        plt.figure()
        plt.title("N vs a for $a_i$ =" + str(self.crack_obj.ai*1000) + " mm")
        plt.plot(a_list*1000, N_list/1000000)
        plt.plot([a_list[-1]*1000,a_list[-1]*1000],[0,N_list[-1]*1.05/1000000])
        plt.text(0.95*1000*a_list[-1],0.2*N_list[-1]/1000000, "Critical Crack length", rotation = 90 )
        plt.text(0.15*1000*a_list[-1], 0.9*N_list[-1]/1000000,"N_f =" + str(np.round(N_list[-1])))
        plt.text(0.5 * 1000 * a_list[-1], 0.9 * N_list[-1]/1000000, "t_f =" + str(np.round(self.crack_obj.t_EOL)) + "hrs")
        plt.xlabel("crack length a [mm]")
        plt.ylabel("Number of cycles x $10^6$ N")

    def Plot_RUL_vs_ai(self):
        EOls = []
        ai_range = np.linspace(0.01e-3, 2e-3, 100)
        for initial_crack_length in ai_range:
            # Operating condition parameters
            ##################
            Pressure_angle = 20
            Number_of_ring_gear_teeth = 62
            Number_of_sun_teeth = 13
            Number_of_planet_gear_teeth = 24
            Module = 2.3e-3  # [m]
            Motor_speed = 1000 # RPM
            opper = Opperating_Conditions(Pressure_angle,
                                          Number_of_ring_gear_teeth,
                                          Number_of_sun_teeth,
                                          Number_of_planet_gear_teeth,
                                          Module,
                                          Motor_speed)

            # Crack parameters
            ##################
            # initial_crack_length = 0.5e-3  # [m]
            tooth_breadth = 5e-3  # [m] From measurement at the tooth root
            tooth_width = 12e-3  # [m]
            diametral_pitch = 1 / Module  # [m]   diametral pitch = 1/module
            Plane_strain_fracture_toughness = 50e6  # [Pa*m^0.5] #6-20
            Paris_Law_C = 1.36e-7  # [(mm/cycle)/(MPa sqrt(m))^m] For martensitic steel
            Paris_Law_M = 2.25  # For martensitic steel
            Lewis_Form_Factor = 0.35  # For 20deg pressure angle, 24 teeth, stub teeth (full involute would be about 0.4)
            Opperating_Condition_object = opper

            crack = Cracked_Gear_Tooth(initial_crack_length,
                                       tooth_breadth,
                                       tooth_width,
                                       diametral_pitch,
                                       Plane_strain_fracture_toughness,
                                       Paris_Law_C,
                                       Paris_Law_M,
                                       Lewis_Form_Factor,
                                       Opperating_Condition_object)

            EOls.append(crack.t_EOL)

        plt.figure()
        plt.title("RUL vs initial crack length")
        plt.plot(ai_range * 1000, EOls)
        plt.xlabel("Initial crack length [mm]")
        plt.ylabel("Time to EOL (RUL) [hours]")
        plt.text(0,1000,"Opperating conditions for this plot is set in Plot_RUL_vs_ai function")




# Operating condition parameters
##################
Pressure_angle = 20
Number_of_ring_gear_teeth = 62
Number_of_sun_teeth = 13
Number_of_planet_gear_teeth = 24
Module = 2.3e-3 #[m] Module of 2.3mm is very uncommon. Usually 2.5. However, this is what I measured
Motor_speed  = 1000 #RPM
opper = Opperating_Conditions(Pressure_angle,
                              Number_of_ring_gear_teeth,
                              Number_of_sun_teeth,
                              Number_of_planet_gear_teeth,
                              Module,
                              Motor_speed)

# Crack parameters
##################
initial_crack_length = 1.21e-3  # [m]
tooth_breadth = 5e-3  # [m] From measurement at the tooth root
tooth_width = 12e-3  #[m]
diametral_pitch = 1/Module #[m]   diametral pitch = 1/module
Plane_strain_fracture_toughness = 50e6 #[Pa*m^0.5] #6-20 could be closer to 50-100
Paris_Law_C = 1.36e-7 # [(mm/cycle)/(MPa sqrt(m))^m] For martensitic steel from Dowling
Paris_Law_M = 2.25    # For martensitic steel
Lewis_Form_Factor = 0.35 # For 20deg pressure angle, 24 teeth, stub teeth (full involute would be about 0.4)
Opperating_Condition_object = opper

crack = Cracked_Gear_Tooth(initial_crack_length,
                           tooth_breadth,
                           tooth_width,
                           diametral_pitch,
                           Plane_strain_fracture_toughness,
                           Paris_Law_C,
                           Paris_Law_M,
                           Lewis_Form_Factor,
                           Opperating_Condition_object)


#Plots
#######################3
plt_obj = Plots(crack,opper)
#plt_obj.Plot_Brittle_Fracture()
#plt_obj.Plot_Torque_vs_speed()
plt_obj.Plot_N_vs_a()
#plt_obj.Plot_RUL_vs_ai()
#
#
# print("Input torque:", opper.T,"[Nm]")
# print("Force on gear tooth:",opper.Planet_tooth_force,"[N]")
# print("EOL", crack.t_EOL, "[hours]")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

crack_range = np.linspace(0,4e-3,100)

plt.figure()
plt.plot(crack_range,crack.Stess_Intensity(crack_range))


"""
# Conclusions

#1) Trade of in terms of operating condition. 
    Too low torque might lead to no crack growth (Stress intensity range below K_th). 
    Too high torque will lead to a shorter critical crack length before brittle fracture. 


initial crack length should be
16hrs : 1.21mm
24hrs : 0.89mm
32hrs : 0.66mm
"""

