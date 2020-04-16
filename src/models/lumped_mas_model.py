# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:02:01 2019

@author: douwm
"""

import numpy as np
import scipy.signal as s
import matplotlib.pyplot as plt
import scipy.integrate as inter
import scipy as sci

class M(object):
    """
    This class is used to create mass matrix objects
    """

    def __init__(self, PG_obj):
        """Initializes the mass matrix object

        Parameters
        ----------
        PG_obj: A Planetary gearbox object
            """

        self.N = PG_obj.N
        self.M_atr = PG_obj.M_atr
        self.matrix_shape = PG_obj.matrix_shape

        self.M_mat = self.M()  # Let Mass object have attribute mass matrix

    def M_j(self, j):
        """
        A single 3x3 component on the diagnonal of the mass matrix

        Parameters
        ----------
        j   :  j = c,r,s,1,2,3... N

        Returns
        -------
        M_j   : 3x3 numpy array

        """
        Mj = np.diag([self.M_atr[0, j], self.M_atr[0, j], self.M_atr[1, j]])

        return Mj

    def M(self):
        """
        Assembles the mass matrix from M_j elements

        Returns
        -------
        M    :3x(3+N) x 3x(3+N) numpy array
        """
        M = np.zeros((self.matrix_shape, self.matrix_shape))

        for j in range(3 + self.N): # For (carrier, ring sun) and planets
            lb = 3 * j
            ub = 3 * (j + 1)
            M[lb:ub, lb:ub] = self.M_j(j)
        return M


class G(object):
    """
    This class is used to create gyroscopic matrix objects
    """

    def __init__(self, PG_obj):
        """Initializes the gyroscopic matrix object

        Parameters
        ----------
        PG_obj: A Planetary gearbox object
            """

        self.N = PG_obj.N
        self.M_atr = PG_obj.M_atr
        self.matrix_shape = PG_obj.matrix_shape

        self.G_mat = self.G()  # Let gyroscopic object have attribute gyroscopic matrix

    def G_j(self, j):
        """
        A single 3x3 component on the diagnonal of the gyroscopic matrix

        Parameters
        ----------
        j   :  j = c,r,s,1,2,3... N

        Returns
        -------
        G_j   : 3x3 numpy array

        """
        G_j = np.zeros((3, 3))
        G_j[0, 1] = -2 * self.M_atr[0, j]
        G_j[1, 0] = 2 * self.M_atr[0, j]

        return G_j

    def G(self):
        """
        Assembles the gyroscopic matrix from G_j elements

        Returns
        -------
        G    :3x(3+p) x 3x(3+p) numpy array
        """
        G = np.zeros((self.matrix_shape, self.matrix_shape))
        for j in range(3 + self.N):
            lb = 3 * j
            ub = 3 * (j + 1)
            G[lb:ub, lb:ub] = self.G_j(j)
        return G


class K_Omega(object):
    """
    This class is used to create gyroscopic matrix objects
    """

    def __init__(self, PG_obj):
        """Initializes the gyroscopic matrix object

        Parameters
        ----------
        PG_obj: A Planetary gearbox object
            """

        self.N = PG_obj.N
        self.M_atr = PG_obj.M_atr
        self.matrix_shape = PG_obj.matrix_shape

        self.K_Omega_mat = self.K_Omega()  # Let gyroscopic object have attribute gyroscopic matrix

    def K_Omega_j(self, j):
        '''
        A single 3x3 component on the diagonal of the gyroscopic matrix

        Parameters
        ----------
        j   :  j = c,r,s,1,2,3... N

        Returns
        -------
        G_j   : 3x3 numpy array

        '''
        K_Omega_j = np.zeros((3, 3))
        K_Omega_j[0, 0] = self.M_atr[0, j]
        K_Omega_j[1, 1] = self.M_atr[0, j]

        return K_Omega_j

    def K_Omega(self):
        '''
        Assembles the gyroscopic matrix from G_j elements

        Returns
        -------
        G    :3x(3+p) x 3x(3+p) numpy array
        '''
        K_Omega = np.zeros((self.matrix_shape, self.matrix_shape))
        for j in range(3 + self.N):
            lb = 3 * j
            ub = 3 * (j + 1)
            K_Omega[lb:ub, lb:ub] = self.K_Omega_j(j)
        return K_Omega


class K_b(object):
    """
    This class is used to create bearing stiffness matrix objects
    """

    def __init__(self, PG_obj):
        """Initializes the bearing stiffness matrix object

        Parameters
        ----------
        PG_obj: A Planetary gearbox object
            """

        self.k_p = PG_obj.k_atr[2] # The bearing stiffness taken for all bearings
        self.kru = PG_obj.k_atr[3]
        self.PG = PG_obj
        self.K_b_mat = self.K_b()

    def K_jb(self, gear):
        '''
        A single 3x3 component on the diagonal of the bearing stiffness matrix

        Parameters
        ----------
        j   :  j = c,r,s

        Returns
        -------
        K_jb   : 3x3 numpy array

        '''
        K_jb = np.zeros((3, 3))
        K_jb[0, 0] = self.k_p
        K_jb[1, 1] = self.k_p

        #if gear == "ring":
        if gear == "ring" or "carrier":
            K_jb[2, 2] = self.kru  # The ring resists rotational motion

        else:
            K_jb[2, 2] = 0         # Planet and sun gears are free to rotate

        return K_jb

    def K_b(self):
        '''
        Assembles the gyroscopic matrix from K_jb elements

        Returns
        -------
        G    :3x(3+N) x 3x(3+N) numpy array
        '''
        K_b = np.zeros((self.PG.N*3+9, self.PG.N*3+9))

        K_b[0:3, 0:3] = self.K_jb("carrier")
        K_b[3:6, 3:6] = self.K_jb("ring")
        K_b[6:9, 6:9] = self.K_jb("sun")

        return K_b


class K_e(object):
    """
    This class is used to create time varying gear mesh stiffness matrix objects
    """

    def __init__(self, PG_obj):
        """Initializes the time varying mesh stiffness matrix object

        Parameters
        ----------
        PG_obj: A Planetary gearbox object
            """

        self.PG = PG_obj
        self.k_atr = PG_obj.k_atr
        self.K_e_mat = self.Compiled

    def k_sp(self,t):
        """
        Time varying mesh stiffness of sun-planet mesh

        Parameters
        ----------
        t:  float
            time

        Returns
        -------
        k_sp: float
              The sun-planet mesh stiffness at a specific point in time
        """
        GMF_sp = 100
        return self.k_atr[0] + self.k_atr[0]*0.5*(s.square(t*2*np.pi*GMF_sp, 0.7)+1)
        #return self.k_atr[0]

    def k_rp(self,t):
        """
        Time varying mesh stiffness of ring-planet mesh

        Parameters
        ----------
        t:  float
            time

        Returns
        -------
        k_sp: float
              The sun-planet mesh stiffness at a specific point in time
        """
        #GMF_rp = 300
        #return self.k_atr[1] + self.k_atr[1]*0.5*(s.square(t*2*np.pi*GMF_rp, 0.7)+1) #Note that the duty cycle is set like this now
        return self.k_atr[1]

    def Kp_s2(self, p, t):
        """
        K^p_s2 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        t   :  float
               Time [s]

        Returns
        -------
        Kp_s2   : 3x3 numpy array
        """

        phi_sp = self.PG.phi_sp_list[p-1]  # -1 due to zero based indexing planets numbered from 1 -> N
        alpha_s = self.PG.alpha_s

        Kp_s2 = np.zeros((3, 3))

        Kp_s2[0, 0] = np.sin(phi_sp)*np.sin(alpha_s)
        Kp_s2[0, 1] = np.sin(phi_sp) * np.cos(alpha_s)
        Kp_s2[0, 2] = -np.sin(phi_sp)

        Kp_s2[1, 0] = -np.cos(phi_sp)*np.sin(alpha_s)
        Kp_s2[1, 1] = -np.cos(phi_sp) * np.cos(alpha_s)
        Kp_s2[1, 2] = np.cos(phi_sp)   #This term is negative in Chaari but positive in Parker 1999

        Kp_s2[2, 0] = -np.sin(alpha_s)
        Kp_s2[2, 1] = -np.cos(alpha_s)
        Kp_s2[2, 2] = 1

        Kp_s2 = self.k_sp(t)*Kp_s2
        return Kp_s2

    def Kp_r2(self, p, t):
        """
        K^p_r2 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        t   :  float
               Time [s]

        Returns
        -------
        Kp_r2   : 3x3 numpy array
        """

        phi_rp = self.PG.phi_rp_list[p-1]  # -1 due to zero based indexing planets numbered from 1 -> N
        alpha_r = self.PG.alpha_r

        Kp_r2 = np.zeros((3, 3))

        Kp_r2[0, 0] = -np.sin(phi_rp)*np.sin(alpha_r)
        Kp_r2[0, 1] = np.sin(phi_rp) * np.cos(alpha_r)
        Kp_r2[0, 2] = np.sin(phi_rp)

        Kp_r2[1, 0] = np.cos(phi_rp)*np.sin(alpha_r)
        Kp_r2[1, 1] = -np.cos(phi_rp) * np.cos(alpha_r)
        Kp_r2[1, 2] = -np.cos(phi_rp)

        Kp_r2[2, 0] = np.sin(alpha_r)
        Kp_r2[2, 1] = -np.cos(alpha_r)
        Kp_r2[2, 2] = -1

        Kp_r2 = self.k_rp(t)*Kp_r2
        return Kp_r2

    def Kp_c2(self, p, t):
        """
        K^p_c2 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        Returns
        -------
        Kp_c2   : 3x3 numpy array
        """

        phi_p = self.PG.phi_p_list [p-1]  # -1 due to zero based indexing planets numbered from 1 -> N

        Kp_c2 = np.zeros((3, 3))
        Kp_c2[0, 0] = -np.cos(phi_p)
        Kp_c2[0, 1] = np.sin(phi_p)
        Kp_c2[0, 2] = 0

        Kp_c2[1, 0] = -np.sin(phi_p)
        Kp_c2[1, 1] = -np.cos(phi_p)
        Kp_c2[1, 2] = 0

        Kp_c2[2, 0] = 0
        Kp_c2[2, 1] = -1
        Kp_c2[2, 2] = 0

        Kp_c2 = self.k_atr[2]*Kp_c2 #  Multiply the matrix with K_p
        return Kp_c2

    def Kp_s1(self, p, t):
        """
        K^p_s1 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        t   :  float
               Time [s]

        Returns
        -------
        Kp_s1   : 3x3 numpy array
        """

        phi_sp = self.PG.phi_sp_list[p-1]  # -1 due to zero based indexing planets numbered from 1 -> N

        Kp_s1 = np.zeros((3, 3))

        Kp_s1[0, 0] = np.sin(phi_sp)**2
        Kp_s1[0, 1] = -np.cos(phi_sp) * np.sin(phi_sp)
        Kp_s1[0, 2] = -np.sin(phi_sp)

        Kp_s1[1, 0] = -np.cos(phi_sp)*np.sin(phi_sp)
        Kp_s1[1, 1] = np.cos(phi_sp)**2
        Kp_s1[1, 2] = np.cos(phi_sp)

        Kp_s1[2, 0] = -np.sin(phi_sp)
        Kp_s1[2, 1] = np.cos(phi_sp)
        Kp_s1[2, 2] = 1

        Kp_s1 = self.k_sp(t)*Kp_s1
        return Kp_s1

    def Kp_r1(self, p, t):
        """
        K^p_r1 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        t   :  float
               Time [s]

        Returns
        -------
        Kp_r1   : 3x3 numpy array
        """

        phi_rp = self.PG.phi_rp_list[p-1]  # -1 due to zero based indexing planets numbered from 1 -> N
        alpha_r = self.PG.alpha_r

        Kp_r1 = np.zeros((3, 3))

        Kp_r1[0, 0] = np.sin(phi_rp)**2
        Kp_r1[0, 1] = -np.cos(phi_rp) * np.sin(phi_rp)# in Parker #-np.cos(phi_rp) * np.cos(alpha_r)
        Kp_r1[0, 2] = -np.sin(phi_rp)   # This term is positvie in Chaari and negative in Parker

        Kp_r1[1, 0] = -np.cos(phi_rp) * np.sin(phi_rp) #In Parker
        Kp_r1[1, 1] = np.cos(phi_rp)**2
        Kp_r1[1, 2] = np.cos(phi_rp)

        Kp_r1[2, 0] = -np.sin(phi_rp) # This term is positvie in Chaari and negative in Parker
        Kp_r1[2, 1] = np.cos(phi_rp)
        Kp_r1[2, 2] = 1

        Kp_r1 = self.k_rp(t)*Kp_r1
        return Kp_r1

    def Kp_c1(self, p, t):
        """
        K^p_c1 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        Returns
        -------
        Kp_c1   : 3x3 numpy array
        """

        phi_p = self.PG.phi_p_list [p-1]  # -1 due to zero based indexing planets numbered from 1 -> N

        Kp_c1 = np.zeros((3, 3))

        Kp_c1[0, 0] = 1
        Kp_c1[0, 1] = 0
        Kp_c1[0, 2] = -np.sin(phi_p)

        Kp_c1[1, 0] = 0
        Kp_c1[1, 1] = 1
        Kp_c1[1, 2] = np.cos(phi_p)

        Kp_c1[2, 0] = -np.sin(phi_p)
        Kp_c1[2, 1] = np.cos(phi_p)
        Kp_c1[2, 2] = 1

        Kp_c1 = self.k_atr[2]*Kp_c1 #  Multiply the matrix with K_p
        return Kp_c1

    def Kp_r3(self, p, t):
        """
        K^p_r3 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        t   :  float
               Time [s]

        Returns
        -------
        Kp_r3   : 3x3 numpy array
        """

        alpha_r = self.PG.alpha_r

        Kp_r3 = np.zeros((3, 3))

        Kp_r3[0, 0] = np.sin(alpha_r)**2
        Kp_r3[0, 1] = -np.cos(alpha_r) * np.sin(alpha_r)
        Kp_r3[0, 2] = -np.sin(alpha_r)

        Kp_r3[1, 0] = -np.cos(alpha_r)*np.sin(alpha_r)
        Kp_r3[1, 1] = np.cos(alpha_r)**2
        Kp_r3[1, 2] = np.cos(alpha_r)

        Kp_r3[2, 0] = -np.sin(alpha_r)
        Kp_r3[2, 1] = np.cos(alpha_r)
        Kp_r3[2, 2] = 1

        Kp_r3 = self.k_rp(t)*Kp_r3
        return Kp_r3

    def Kp_s3(self, p, t):
        """
        K^p_s3 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        t   :  float
               Time [s]

        Returns
        -------
        Kp_s3   : 3x3 numpy array
        """

        alpha_s = self.PG.alpha_s

        Kp_s3 = np.zeros((3, 3))

        Kp_s3[0, 0] = np.sin(alpha_s)**2
        Kp_s3[0, 1] = np.cos(alpha_s) * np.sin(alpha_s)
        Kp_s3[0, 2] = -np.sin(alpha_s)

        Kp_s3[1, 0] = np.cos(alpha_s)*np.sin(alpha_s)
        Kp_s3[1, 1] = np.cos(alpha_s)**2
        Kp_s3[1, 2] = -np.cos(alpha_s)

        Kp_s3[2, 0] = -np.sin(alpha_s)
        Kp_s3[2, 1] = -np.cos(alpha_s)
        Kp_s3[2, 2] = 1

        Kp_s3 = self.k_sp(t)*Kp_s3
        return Kp_s3

    def Kp_c3(self, p, t ):
        """
        K^p_c3 component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number

        Returns
        -------
        Kp_c3   : 3x3 numpy array
        """

        Kp_c3 = np.zeros((3, 3))

        Kp_c3[0, 0] = 1
        Kp_c3[0, 1] = 0
        Kp_c3[0, 2] = 0

        Kp_c3[1, 0] = 0
        Kp_c3[1, 1] = 1
        Kp_c3[1, 2] = 0

        Kp_c3[2, 0] = 0
        Kp_c3[2, 1] = 0
        Kp_c3[2, 2] = 0

        Kp_c3 = self.k_atr[2]*Kp_c3#  Multiply the matrix with K_p
        return Kp_c3

    def Kp(self, p, t):
        """
        K^p component of mesh stiffness matrix

        Parameters
        ----------
        p   :  int
               The planet gear number
        t   :  float
               Time [s]

        Returns
        -------
        Kp   : 3x3 numpy array
        """
        Kp = self.Kp_c3(p,t) + self.Kp_r3(p, t) + self.Kp_s3(p, t)
        return Kp

    def Sum_Kp_c1(self, t):
        """
        Sum of all K^p_c1 over all p (planets) for mesh stiffness matrix

        Parameters
        ----------
        t   :  float
               Time

        Returns
        -------
        Sum_Kp_c1   : 3x3 numpy array
        """
        Sum_Kp_c1 = 0
        for p in range(1,self.PG.N+1):
            Sum_Kp_c1 += self.Kp_c1(p, t)
        return Sum_Kp_c1

    def Sum_Kp_r1(self, t):
        """
        Sum of all K^p_r1 over all p (planets) for mesh stiffness matrix

        Parameters
        ----------
        t   :  float
               Time

        Returns
        -------
        Sum_Kp_r1   : 3x3 numpy array
        """
        Sum_Kp_r1 = 0
        for p in range(1, self.PG.N+1):
            Sum_Kp_r1 += self.Kp_r1(p, t)
        return Sum_Kp_r1

    def Sum_Kp_s1(self, t):
        """
        Sum of all K^p_s1 over all p (planets) for mesh stiffness matrix

        Parameters
        ----------
        t   :  float
               Time

        Returns
        -------
        Sum_Kp_s1   : 3x3 numpy array
        """
        Sum_Kp_s1 = 0
        for p in range(1, self.PG.N+1):
            Sum_Kp_s1 += self.Kp_s1(p, t)
        return Sum_Kp_s1

    def Off_Diag(self, t):
        """
        Creates off diagonal rectangular sections (lower left) for stiffness matrix

        Parameters
        ----------
        t   :  float
               Time

        Returns
        -------
        Cols   : (3xN)x9 numpy array
        """
        #Cols = np.zeros((3*self.PG.N, 9))

        #for p in range(1,self.PG.N+1):
            #Cols[3 * (p - 1):3 * (p +1 - 1), 0:3] = self.Kp_c2(p, t)
            #Cols[3 * (p - 1):3 * (p + 1 - 1), 3:6] = self.Kp_r2(p, t)
            #Cols[3 * (p - 1):3 * (p + 1 - 1), 6:9] = self.Kp_s2(p, t)

        #return Cols

        rect = np.zeros((9,self.PG.N*3))

        for p in range(1,self.PG.N +1):
            rect[0:3,(p-1)*3:p*3] = self.Kp_c2(p, t)
            rect[3:6, (p - 1) * 3:p * 3] = self.Kp_r2(p, t)
            rect[6:9, (p - 1) * 3:p * 3] = self.Kp_s2(p, t)

        return rect

    def Right_Bottom(self, t):
        """
        Creates a square matrix from 3x3 K^p matrices for the stiffness matrix

        Parameters
        ----------
        t   :  float
               Time

        Returns
        -------
        square   : (3xN)x(3xN) numpy array
                N is the number of planet gears
        """
        square = np.zeros((3*self.PG.N, 3*self.PG.N))

        for p in range(1,self.PG.N+1):
            square[3*(p-1):3*(p+1-1), 3*(p-1):3*(p-1+1)] = self.Kp(p, t)
        return square

    def Left_Top(self, t):
        """
        Creates a square matrix from 3x3 sum of K^p_gear1 matrices for the stiffness matrix

        Parameters
        ----------
        t   :  float
               Time

        Returns
        -------
        square   : 9x9 numpy array
        """
        square = np.zeros((9, 9))

        square[0:3, 0:3] = self.Sum_Kp_c1(t)
        square[3:6, 3:6] = self.Sum_Kp_r1(t)
        square[6:9, 6:9] = self.Sum_Kp_s1(t)

        return square

    def Compiled(self,t):
        """
        Creates a square matrix Ke(t)

        Parameters
        ----------
        t   :  float
               Time

        Returns
        -------
        Ke   : (9+3xN)x(9+3xN) numpy array
                Time varying stiffness matrix
        """
        Ke = np.zeros((9 + 3*self.PG.N, 9 + 3*self.PG.N))
        Ke[0:9,0:9] = self.Left_Top(t)
        Ke[9:, 9:] = self.Right_Bottom(t)

        #off_diag = self.Off_Diag(t)
        #Ke[9:, 0:9] = off_diag
        #Ke[0:9, 9:] = off_diag.T

        off_diag = self.Off_Diag(t)
        Ke[9:, 0:9] = off_diag.T
        Ke[0:9, 9:] = off_diag
        return Ke


class T(object):
    """
    This class is used to create torque vector objects
    """

    def __init__(self, PG_obj):
        """Initializes the torque vector object

        Parameters
        ----------
        PG_obj: A Planetary gearbox object
            """

        self.T_s = PG_obj.T_s
        self.N = PG_obj.N

        self.T_vec = self.T()  # Let Mass object have attribute mass matrix

    def T(self):
        """
        Calculates the torque vector

        Parameters
        ----------


        Returns
        -------
        T   : (9+3xN) x 1  numpy array

        """
        T_vec = np.zeros((9+3*self.N, 1))

        #T_vec[2, 0] = -(1+70/30)*self.T_s
        T_vec[8, 0] = self.T_s # Sun

        return T_vec


class DE_Integration(object):
    """
    This class is used to create mass matrix objects
    """

    def __init__(self, PG_obj):
        """Initializes the DE integration object

        Parameters
        ----------
        PG_obj: A Planetary gearbox object
            """

        self.PG = PG_obj

    def E_Q(self,t):
        """
        Converts the second order differential equation to first order (E matrix and Q vector)

        Parameters
        ----------
        t  : Float
             Time

        Returns
        -------
        E  : 2x(9+3xN) x 2x(9+3xN) Numpy array

        Based on Runge-kutta notes

        """
        m = self.PG.M
        k = self.PG.K_b + self.PG.K_e(t) - self.PG.Omega_c**2 * self.PG.K_Omega


        c = self.PG.Omega_c*self.PG.G +  (0.03*m + 0.03*k)  # 0.03*m +0.03*k is proportional damping used to ensure that
                                                        # that the DE integration converges
        F = self.PG.T

        #convert to units 1kN = 1 g . micro_m / micro_s^2,
        # 1kg * 1e3 = g
        # 1N/m * 1e-9 = N/nano_m,
        # 1N/(m/s) * 1e-3 = N/(nano_m/micro_s)
        # 1N * 1 = N

        #m = m/1E3
        #k = k/1E9
        #c = c/1E3
        #F = F

        c_over_m = np.linalg.solve(m, c)
        k_over_m = np.linalg.solve(m, k)
        F_over_m = np.linalg.solve(m, F)

        dim = 2*(9+3*self.PG.N) # Matrix dimension
        half_dim = int(dim/2)

        E = np.zeros((dim, dim))
        E[half_dim:, 0:half_dim] = -k_over_m
        E[half_dim:, half_dim:] = -c_over_m
        E[0:half_dim, half_dim:] = np.eye(half_dim)

        Q = np.zeros((dim, 1))
        Q[half_dim:, 0] = F_over_m[:,0]

        return E, Q

    def X_dot(self, X, t):

        #Xk = np.zeros(6)

        E, Q = self.E_Q(t)
        #Euk, Euu, Qu = self.Prepare_E_and_Q(E, Q)

        X_dot = np.dot(E, np.array([X]).T) + Q

        #X_dot = np.dot(Euk, np.array([Xk]).T) + np.dot(Euu, np.array([Xu]).T) + Qu

        return(X_dot[:,0])

    def Prepare_E_and_Q(self, E, Q):

        """Rearanges the equations to have known values at top of matrix"""
        dim = 2*(9+3*self.PG.N) # Matrix dimension
        half_dim = int(dim/2)

        E_known_at_top = E[0:3,:]
        E_known_at_top = np.vstack((E_known_at_top, E[half_dim:half_dim+3, :]))
        E_known_at_top = np.vstack((E_known_at_top, E[3:half_dim, :]))
        E_known_at_top = np.vstack((E_known_at_top, E[half_dim+3:, :]))

        Q_known_at_top = Q[0:3,:]
        Q_known_at_top = np.vstack((Q_known_at_top, Q[half_dim:half_dim+3, :]))
        Q_known_at_top = np.vstack((Q_known_at_top, Q[3:half_dim, :]))
        Q_known_at_top = np.vstack((Q_known_at_top, Q[half_dim+3:, :]))

        Euk = E_known_at_top[6:, 0:6]
        Euu = E_known_at_top[6:, 6:]

        Qu = Q_known_at_top[6:,:]

        return Euk, Euu, Qu

    def X_0(self):
        """
        Used to easily set up the initial conditions for differntial equation integration

        Parameters
        ----------


        Returns
        -------
        X_0  : 2x(9+3xN) x 1 numpy array
               Initial conditions

        """
        dim = 2 * (9 + 3 * self.PG.N)  # Matrix dimension
        #X_0 = np.zeros((dim, 1))
        X_0 = np.zeros(dim)

        #X_0[6] = 0.00000001
        #X_0[5] = 0.00000001

        #for i in range(self.PG.N):
         #   x, y = self.Initialplanet(i+1)
         #   X_0[6+3*i] = x
          #  X_0[6+3*i+1] = y
          #  X_0[6+3*i+1] = 0.016*self.PG.phi_p_list[i]

        return X_0

    def Run_Integration(self, X_0, t):
        sol = inter.odeint(self.X_dot, X_0, t)#,full_output=1)
        return sol


class Planetary_Gear(object):

    def __init__(self, N, M_atr, Geom_atr,k_atr,Opp_atr):
        """Initializes the planetary gear  object

        Parameters
        ----------
        N : int
            N is the number of planet gears.

        M_atr : 2x(3+N) numpy array
            Mass attribute matrix
            row 0 contains the m_j terms
            row 1 contains the I_j/r^2_j terms

        Geom_atr : 1x2 numpy array
            Geometrical attribute matrix
            np.array(alpha_S, alpha_r)
            """

        self.N = N
        self.M_atr = M_atr

        self.Omega_c = Opp_atr[0]  # Constant carrier rotational speed
        self.T_s = Opp_atr[1]      # Constant torque applied to sun gear

        self.alpha_s = Geom_atr[0]
        self.alpha_r = Geom_atr[1]

        self.k_atr = k_atr        # The stiffness attributes of the planetary gearbox

        if np.shape(M_atr)[1]-3 != self.N:
            raise ValueError("Number of planet gears not in agreement with Mass attribute array size")

        self.matrix_shape = 3 * (3 + self.N)  # The dimension of matrices such as M,G,K

        self.phi_p_list = self.phi_p(np.arange(1, self.N+1)) # A list of the phi_p values for planet gears 1 -> N
        self.phi_sp_list = self.phi_sp(self.phi_p_list)  # A list of the phi_sp values for planet gears 1 -> N
        self.phi_rp_list = self.phi_rp(self.phi_p_list)  # A list of the phi_rp values for planet gears 1 -> N

        # Construct all of the matrices required for the equation of motion
        self.M = M(self).M_mat
        self.G = G(self).G_mat
        self.K_b = K_b(self).K_b_mat
        self.K_e = K_e(self).Compiled  # This is a function. Takes the argument t [s]
        self.K_Omega = K_Omega(self).K_Omega_mat
        self.T = T(self).T_vec

    def phi_p(self, p):
        """
        Determines the circumferential planet positions phi_p

        Parameters
        ----------
        p   :  int
               The planet gear number

        Returns
        -------
        phi_p   : float
                  The circumferential planet position in radians
        """
        increment= 2*np.pi/self.N  # The angular distance between planet gears
        return increment*(p-1)

    def phi_sp(self, phi_p):
        out = phi_p - self.alpha_s
        return out

    def phi_rp(self, phi_p):
        out = phi_p + self.alpha_r
        return out

