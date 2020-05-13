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
import time
import src.features.second_order_solvers as solvers


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

        for j in range(3 + self.N):  # For (carrier, ring sun) and planets
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

        self.k_p = PG_obj.k_atr[2]  # The bearing stiffness taken for all bearings
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

        # if gear == "ring":
        if gear == "ring":
            K_jb[2, 2] = self.kru  # The ring resists rotational motion

        if gear == "carrier":
            K_jb[2, 2] = self.kru

        else:
            K_jb[2, 2] = 0  # sun gear free to rotate

        return K_jb

    def K_b(self):
        '''
        Assembles the gyroscopic matrix from K_jb elements

        Returns
        -------
        G    :3x(3+N) x 3x(3+N) numpy array
        '''
        K_b = np.zeros((self.PG.N * 3 + 9, self.PG.N * 3 + 9))

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
        self.Omega_c = PG_obj.Omega_c

    def k_sp(self, t):
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
        return self.k_atr[0] + self.k_atr[0] * 0.5 * (s.square(t * 2 * np.pi * GMF_sp, 0.5) + 1)
        # return self.k_atr[0] + self.k_atr[0] * 0.5 * (np.sin(t * 2 * np.pi * GMF_sp) + 1)
        # return self.k_atr[0] + t*0
        # return self.k_atr[0]

    def k_rp(self, t):
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
        GMF_rp = 100
        return self.k_atr[1] + self.k_atr[1] * 0.5 * (
                s.square(t * 2 * np.pi * GMF_rp, 0.5) + 1)  # Note that the duty cycle is set like this now
        # return self.k_atr[0] + self.k_atr[0] * 0.5 * (np.sin(t * 2 * np.pi * GMF_rp) + 1)
        # return self.k_atr[1] + t*0
        # return self.k_atr[1]

        # A = 1;
        # f = 2;
        # smoothsq = (2 * A / pi) * atan(sin(2 * pi * t * f) / self.PG. );
        # plot(t, smoothsq);
        # axis([-0.2 2.2 - 1.6 1.6]);

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

        phi_sp = self.PG.phi_sp_list[p - 1]  # -1 due to zero based indexing planets numbered from 1 -> N
        alpha_s = self.PG.alpha_s

        Kp_s2 = np.zeros((3, 3))

        Kp_s2[0, 0] = np.sin(phi_sp) * np.sin(alpha_s)
        Kp_s2[0, 1] = np.sin(phi_sp) * np.cos(alpha_s)
        Kp_s2[0, 2] = -np.sin(phi_sp)

        Kp_s2[1, 0] = -np.cos(phi_sp) * np.sin(alpha_s)
        Kp_s2[1, 1] = -np.cos(phi_sp) * np.cos(alpha_s)
        Kp_s2[1, 2] = np.cos(phi_sp)  # This term is negative in Chaari but positive in Parker 1999

        Kp_s2[2, 0] = -np.sin(alpha_s)
        Kp_s2[2, 1] = -np.cos(alpha_s)
        Kp_s2[2, 2] = 1

        Kp_s2 = self.k_sp(t) * Kp_s2
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

        phi_rp = self.PG.phi_rp_list[p - 1]  # -1 due to zero based indexing planets numbered from 1 -> N
        alpha_r = self.PG.alpha_r

        Kp_r2 = np.zeros((3, 3))

        Kp_r2[0, 0] = -np.sin(phi_rp) * np.sin(alpha_r)
        Kp_r2[0, 1] = np.sin(phi_rp) * np.cos(alpha_r)
        Kp_r2[0, 2] = np.sin(phi_rp)

        Kp_r2[1, 0] = np.cos(phi_rp) * np.sin(alpha_r)
        Kp_r2[1, 1] = -np.cos(phi_rp) * np.cos(alpha_r)
        Kp_r2[1, 2] = -np.cos(phi_rp)

        Kp_r2[2, 0] = np.sin(alpha_r)
        Kp_r2[2, 1] = -np.cos(alpha_r)
        Kp_r2[2, 2] = -1

        Kp_r2 = self.k_rp(t) * Kp_r2
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

        phi_p = self.PG.phi_p_list[p - 1]  # -1 due to zero based indexing planets numbered from 1 -> N

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

        Kp_c2 = self.k_atr[2] * Kp_c2  # Multiply the matrix with K_p
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

        phi_sp = self.PG.phi_sp_list[p - 1]  # -1 due to zero based indexing planets numbered from 1 -> N

        Kp_s1 = np.zeros((3, 3))

        Kp_s1[0, 0] = np.sin(phi_sp) ** 2
        Kp_s1[0, 1] = -np.cos(phi_sp) * np.sin(phi_sp)
        Kp_s1[0, 2] = -np.sin(phi_sp)

        Kp_s1[1, 0] = -np.cos(phi_sp) * np.sin(phi_sp)
        Kp_s1[1, 1] = np.cos(phi_sp) ** 2
        Kp_s1[1, 2] = np.cos(phi_sp)

        Kp_s1[2, 0] = -np.sin(phi_sp)
        Kp_s1[2, 1] = np.cos(phi_sp)
        Kp_s1[2, 2] = 1

        Kp_s1 = self.k_sp(t) * Kp_s1
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

        phi_rp = self.PG.phi_rp_list[p - 1]  # -1 due to zero based indexing planets numbered from 1 -> N
        alpha_r = self.PG.alpha_r

        Kp_r1 = np.zeros((3, 3))

        author = "Parker"

        if author == "Parker":
            Kp_r1[0, 0] = np.sin(phi_rp) ** 2
            Kp_r1[0, 1] = -np.cos(phi_rp) * np.sin(phi_rp)  # sin() argument phi_rp in Parker, alpha_rp in Chaari
            Kp_r1[0, 2] = -np.sin(phi_rp)  # This term is positive in Chaari and negative in Parker

            Kp_r1[1, 0] = -np.cos(phi_rp) * np.sin(phi_rp)  # sin() argument phi_rp in Parker, alpha_rp in Chaari
            Kp_r1[1, 1] = np.cos(phi_rp) ** 2
            Kp_r1[1, 2] = np.cos(phi_rp)

            Kp_r1[2, 0] = -np.sin(phi_rp)  # This term is positvie in Chaari and negative in Parker
            Kp_r1[2, 1] = np.cos(phi_rp)
            Kp_r1[2, 2] = 1

        if author == "Chaari":
            Kp_r1[0, 0] = np.sin(phi_rp) ** 2
            Kp_r1[0, 1] = -np.cos(phi_rp) * np.sin(alpha_r)  # in Parker
            Kp_r1[0, 2] = +np.sin(phi_rp)  # This term is positive in Chaari and negative in Parker

            Kp_r1[1, 0] = -np.cos(phi_rp) * np.sin(alpha_r)
            Kp_r1[1, 1] = np.cos(phi_rp) ** 2
            Kp_r1[1, 2] = np.cos(phi_rp)

            Kp_r1[2, 0] = +np.sin(phi_rp)  # This term is positvie in Chaari and negative in Parker
            Kp_r1[2, 1] = np.cos(phi_rp)
            Kp_r1[2, 2] = 1

        Kp_r1 = self.k_rp(t) * Kp_r1
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

        phi_p = self.PG.phi_p_list[p - 1]  # -1 due to zero based indexing planets numbered from 1 -> N

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

        Kp_c1 = self.k_atr[2] * Kp_c1  # Multiply the matrix with K_p
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

        Kp_r3[0, 0] = np.sin(alpha_r) ** 2
        Kp_r3[0, 1] = -np.cos(alpha_r) * np.sin(alpha_r)
        Kp_r3[0, 2] = -np.sin(alpha_r)

        Kp_r3[1, 0] = -np.cos(alpha_r) * np.sin(alpha_r)
        Kp_r3[1, 1] = np.cos(alpha_r) ** 2
        Kp_r3[1, 2] = np.cos(alpha_r)

        Kp_r3[2, 0] = -np.sin(alpha_r)
        Kp_r3[2, 1] = np.cos(alpha_r)
        Kp_r3[2, 2] = 1

        Kp_r3 = self.k_rp(t) * Kp_r3
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

        Kp_s3[0, 0] = np.sin(alpha_s) ** 2
        Kp_s3[0, 1] = np.cos(alpha_s) * np.sin(alpha_s)
        Kp_s3[0, 2] = -np.sin(alpha_s)

        Kp_s3[1, 0] = np.cos(alpha_s) * np.sin(alpha_s)
        Kp_s3[1, 1] = np.cos(alpha_s) ** 2
        Kp_s3[1, 2] = -np.cos(alpha_s)

        Kp_s3[2, 0] = -np.sin(alpha_s)
        Kp_s3[2, 1] = -np.cos(alpha_s)
        Kp_s3[2, 2] = 1

        Kp_s3 = self.k_sp(t) * Kp_s3
        return Kp_s3

    def Kp_c3(self, p, t):
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

        Kp_c3 = self.k_atr[2] * Kp_c3  # Multiply the matrix with K_p
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
        Kp = self.Kp_c3(p, t) + self.Kp_r3(p, t) + self.Kp_s3(p, t)
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
        for p in range(1, self.PG.N + 1):
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
        for p in range(1, self.PG.N + 1):
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
        for p in range(1, self.PG.N + 1):
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
        # Cols = np.zeros((3*self.PG.N, 9))

        # for p in range(1,self.PG.N+1):
        # Cols[3 * (p - 1):3 * (p +1 - 1), 0:3] = self.Kp_c2(p, t)
        # Cols[3 * (p - 1):3 * (p + 1 - 1), 3:6] = self.Kp_r2(p, t)
        # Cols[3 * (p - 1):3 * (p + 1 - 1), 6:9] = self.Kp_s2(p, t)

        # return Cols

        rect = np.zeros((9, self.PG.N * 3))

        for p in range(1, self.PG.N + 1):
            rect[0:3, (p - 1) * 3:p * 3] = self.Kp_c2(p, t)
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
        square = np.zeros((3 * self.PG.N, 3 * self.PG.N))

        for p in range(1, self.PG.N + 1):
            square[3 * (p - 1):3 * (p + 1 - 1), 3 * (p - 1):3 * (p - 1 + 1)] = self.Kp(p, t)
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

    def Compiled(self, t):
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
        Ke = np.zeros((9 + 3 * self.PG.N, 9 + 3 * self.PG.N))
        Ke[0:9, 0:9] = self.Left_Top(t)
        Ke[9:, 9:] = self.Right_Bottom(t)

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

        self.kru = PG_obj.k_atr[3]

        self.PG = PG_obj

        self.M_atr = PG_obj.M_atr

    def T_vec_base_excitation(self, t):
        """
        Calculates the torque vector

        Parameters
        ----------


        Returns
        -------
        T   : (9+3xN) x 1  numpy array

        """
        T_vec = np.zeros((9 + 3 * self.N, 1))

        vb = 0.001  # constant base velocity # In this case the base is the planet carrier
        xb = vb * t  # base displacement
        T_vec[2, 0] = self.kru * xb  # -(1+70/30)*self.T_s
        T_vec[8, 0] = - self.T_s  # Sun

        return T_vec

    def T_vec_stationary(self, t):
        """
        Calculates the torque vector

        Parameters
        ----------


        Returns
        -------
        T   : (9+3xN) x 1  numpy array

        """
        T_vec = np.zeros((9 + 3 * self.N, 1))
        T_vec[2, 0] = 0
        T_vec[8, 0] = self.T_s  # Sun

        return T_vec


class PG_ratios(object):
    """
    Ratios in a planetary gearbox
    At this stage this class is duplicated in proclib.py as PG
    """
    def __init__(self, Z_r, Z_s, Z_p):
        self.Z_r = Z_r
        self.Z_s = Z_s
        self.Z_p = Z_p
        self.GR = self.GR_calc()

        self.carrier_revs_to_repeat = 12  # The number of revolution of the carrier required for a given planet gear
        # tooth to mesh with the same ring gear tooth. This could be calculated based
        # on the input parameters with the meshing sequence function is extended

        self.Mesh_Sequence = self.Meshing_sequence()  # Calculate the meshing sequence

    def GMF1(self, f_sun):
        """Function that returns the gear mesh frequency for a given sun gear rotation speed f_s
        Parameters
        ----------
        f_s: float
            Sun gears shaft frequency


        Returns
        -------
        GMF: Float
            The gear mesh frequency in Hz
            """
        return f_sun*self.Z_r*self.Z_s/(self.Z_r + self.Z_s)

    def GR_calc(self):
        """Gear ratio for planetary gearbox with stationary ring gear
        Parameters
        ----------


        Returns
        -------
        GR: Float
            The ratio
            """
        return 1 + self.Z_r / self.Z_s

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
        return f_c*self.Z_r/self.Z_p

    def GMF(self,f_s):
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
        fc = f_s/self.GR
        return self.f_p(fc)*self.Z_p

    def FF1(self,f_s):
        """Calculates the gear mesh frequency for a given a sun gear input frequency. The gearbox is therefore running in the speed down configuration
        Parameters
        ----------
        f_sun: Float
             Frequency of rotation of sun gear


        Returns
        -------
        FF1: Float
            Fault frequency due to fault on planet gear
            """
        f_c = f_s/self.GR # Calculate the frequency of rotation of the carrier
        fp = self.f_p(f_c)
        FF1 = 2*fp # The fault will manifest in the vibration twice per revolution of the planet gear:
        # once at the sun gear and once at the ring gear
        return FF1

    def Meshing_sequence(self):
        """Determines the order in which the teeth of a planet gear will mesh with the ring gear
        Parameters
        ----------


        Returns
        -------
        Mesh_Sequence: array
            Array of tooth numbers (zero to Np-1) that show the order in which the teeth will mesh
            """

        Mesh_Sequence = []
        for n_rev in range(self.carrier_revs_to_repeat): #Notice that the sequence starts repeating after 12 rotations
            Mesh_Sequence.append((n_rev*self.Z_r)%self.Z_p)

        return Mesh_Sequence

    def fatigue_cycles(self, carrier_revs):
        """
        Calculates the number of fatigue cycles given the number of planet rotations
        Parameters
        ----------
        carrier_revs: float
                    Number of rotations of the carrier

        Returns
        -------
        fatigue_cycles
        """
        return float(carrier_revs)*(self.Z_r/self.Z_p) # (number of revs)(number of cycles per revolution)


    @classmethod
    def RPM_to_f(cls,RPM):
        """Function converts revolutions per minute to Herz
        Parameters
        ----------
        RPM: Float
            Rotational speed in RPM


        Returns
        -------
        f:  Float
            Frequency [Hz]
            """
        return RPM/60

class Transmission_Path(object):
    """
    Models the transmission path of the vibration to the accelerometer

    Based on Parra and Vicuna 2017 eq(2) and modified hamming function from Liang et al 2015 eq(3.1)
    """

    def __init__(self, PG_obj):
        """
        Initialize transmission path object

        solution: n_DOF x n_timesteps array
        """
        self.PG = PG_obj
        self.sol = PG_obj.time_domain_solution.T
        self.time_range = self.PG.time_range

        return

    def y(self):
        summation = 0
        for planet in range(1, self.PG.N + 1):
            summation += self.s_ri(planet) * self.F_ri(planet, self.time_range) \
                         * np.sin(self.PG.Omega_c * self.time_range + self.PG.alpha_r + self.PG.phi_p_list[planet -1 ])
        return summation

    def F_ri(self, planet, t_range):
        return self.PG.k_rp(t_range) * self.d_ri(planet)

    def d_ri(self, planet):
        """
        planet makes use of 1 based (non zero based) indexing
        Parameters
        ----------
        planet: Planet number 1->N

        Returns
        -------

        """
        i = 9 + (planet - 1) * 3  # 9 DOF for sun, ring and carrier. 1 Based indexing for planets

        phi_p = self.PG.phi_p(planet)  # Relative planet angular positions
        alpha_r = self.PG.alpha_r

        t1 = + self.sol[3 + 1, :] * np.cos(phi_p)
        t2 = - self.sol[3, :] * np.sin(phi_p)
        t3 = + self.sol[i + 1, :] * np.sin(alpha_r)
        t4 = - self.sol[i, :] * np.cos(alpha_r)
        t5 = + self.sol[3 + 2, :]
        t6 = - self.sol[i + 2, :]

        return t1 + t2 + t3 + t4 + t5 + t6

    def s_ri(self, planet):
        """
        Accounts for variable transmission path due to movement of planet gears with respect to fixed transducer
        (Parra, Vicuna 2017). This also includes amplitude modulation
        Returns
        -------

        """
        return 1  # No amplitude modulation or variable transmission path. Assume constant

    def window_extract(self, wind_length, y, Omega_c, fs):
        """
        Extracts windows of length l samples every 2*pi/udot_c seconds
        In  other words this extracts a window of samples as a planet gear passes the transducer
         """

        if wind_length % 2 is not 0 == True:
            raise ValueError("Please enter uneven window length")

        samples_per_rev = int((1/2*np.pi)*(1/Omega_c)*fs)
        print("fs ", fs)
        print("samples per rev ", samples_per_rev)
        window_center_index = np.arange(0, len(y), samples_per_rev).astype(int)

        n_revs = np.shape(window_center_index)[0] - 2  # exclude the first and last revolution to prevent errors with insufficient window length
        # first window would have given problems requireing negative sample indexes

        windows = np.zeros((n_revs, wind_length))  # Initialize an empty array that will hold the extracted windows
        window_half_length = int(np.floor(wind_length/2))

        window_count = 0
        for index in window_center_index[1:-1]:  # exclude the first and last revolution to prevent errors with insufficient window length
            windows[window_count, :] = y[index - window_half_length:index + window_half_length + 1]
            window_count += 1
        return windows

class Planetary_Gear(object):

    def __init__(self, N, M_atr, Geom_atr, k_atr, Opp_atr, solve_attr):
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

        self.Omega_c = Opp_atr["Omega_c"]
        self.T_s = Opp_atr["T_s"]
        self.base_excitation = Opp_atr["base_excitation"]

        self.alpha_s = Geom_atr[0]
        self.alpha_r = Geom_atr[1]

        self.k_atr = k_atr  # The stiffness attributes of the planetary gearbox

        if np.shape(M_atr)[1] - 3 != self.N:
            raise ValueError("Number of planet gears not in agreement with Mass attribute array size")

        self.matrix_shape = 3 * (3 + self.N)  # The dimension of matrices such as M,G,K

        self.phi_p_list = self.phi_p(np.arange(1, self.N + 1))  # A list of the phi_p values for planet gears 1 -> N
        self.phi_sp_list = self.phi_sp(self.phi_p_list)  # A list of the phi_sp values for planet gears 1 -> N
        self.phi_rp_list = self.phi_rp(self.phi_p_list)  # A list of the phi_rp values for planet gears 1 -> N

        # Construct all of the matrices required for the equation of motion
        self.M = M(self).M_mat
        self.G = G(self).G_mat
        self.K_b = K_b(self).K_b_mat
        self.K_e = K_e(self).Compiled  # This is a function. Takes the argument t [s]
        self.K_Omega = K_Omega(self).K_Omega_mat

        if self.base_excitation:
            self.T = T(self).T_vec_base_excitation  # This is a function. Takes the argument t [s]
        else:
            self.T = T(self).T_vec_stationary

        self.K = lambda t: self.K_b + self.K_e(t) - self.Omega_c ** 2 * self.K_Omega

        self.solve_atr = solve_attr
        self.time_range = solve_attr["time_range"]
        self.fs = 1/np.average(np.diff(self.time_range))

        # Make proportional damping time dependent or constant
        if self.solve_atr["proportional_damping_constant"]:
            self.C = lambda t: self.Omega_c * self.G + self.solve_atr["time_varying_proportional_damping"] * (
                    self.M + self.K(t))

        else:
            self.C = lambda t: self.Omega_c * self.G + self.solve_atr["time_varying_proportional_damping"] * (
                    self.M + self.K(0))

        self.k_sp = K_e(self).k_sp  # This is a function. Takes the argument t [s]
        self.k_rp = K_e(self).k_rp  # This is a function. Takes the argument t [s]
        return

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
        increment = 2 * np.pi / self.N  # The angular distance between planet gears
        return increment * (p - 1)

    def phi_sp(self, phi_p):
        out = phi_p - self.alpha_s
        return out

    def phi_rp(self, phi_p):
        out = phi_p + self.alpha_r
        return out

    def plot_tvms(self, time_range):
        plt.figure("Ring Planet Stiffness")
        plt.plot(time_range, self.k_rp(time_range))
        plt.xlabel("Time [s]")
        plt.ylabel("Mesh Stiffness [N/m")

        plt.figure("Sun Planet Stiffness")
        plt.plot(time_range, self.k_sp(time_range))
        plt.xlabel("Time [s]")
        plt.ylabel("Mesh Stiffness [N/m")

        return

    def plot_solution(self, state_time_der):

        try:
            nstate = int(np.shape(self.time_domain_solution)[1] / 3)
        except AttributeError:
            print("Please run solution using get_solution method")
            return

        if state_time_der == "Displacement":
            start = 0

        if state_time_der == "Velocity":
            start = nstate * 1

        if state_time_der == "Acceleration":
            start = nstate * 2

        end = start + nstate

        lables = ("x_c",
                  "y_c",
                  "u_c",
                  "x_r",
                  "y_r",
                  "u_r",
                  "x_s",
                  "y_s",
                  "u_s",
                  "zeta_1",
                  "nu_1",
                  "u_1")

        # plt.figure(state_time_der)
        # plt.ylabel("Displacement [m]")
        # plt.xlabel("Time [s]")
        # p = plt.plot(self.time_range[1:], solution[1:, start:end])
        # plt.legend(iter(p), lables)

        plt.figure("Rotational DOF, carrier, sun, planet" + state_time_der)
        plt.plot(self.time_range, self.time_domain_solution[:, start + 0], label="u_c")
        plt.plot(self.time_range, self.time_domain_solution[:, start + 2], label="u_r")
        plt.plot(self.time_range, self.time_domain_solution[:, start + 8], label="u_s")
        plt.plot(self.time_range, self.time_domain_solution[:, start + 11], label="u_1")
        plt.legend()

        plt.figure("x-translation, carrier, sun, planet" + state_time_der)
        plt.plot(self.time_range, self.time_domain_solution[:, start + 0], label="x_c")
        plt.plot(self.time_range, self.time_domain_solution[:, start + 3], label="x_r")
        plt.plot(self.time_range, self.time_domain_solution[:, start + 6], label="x_s")
        plt.plot(self.time_range, self.time_domain_solution[:, start + 9], label="x_p1")
        plt.legend()

        plt.figure("Planet displacement")
        plt.plot(self.time_range, self.time_domain_solution[:, start + 9], label="zeta_1")
        plt.plot(self.time_range, self.time_domain_solution[:, start + 10], label="nu_1")
        plt.legend()

    def get_solution(self):
        try:
            return self.time_domain_solution
        except AttributeError:
            self.time_domain_solution = self.run_integration()
            return self.time_domain_solution

    def run_integration(self):

        X0 = self.solve_atr["X0"]
        Xd0 = self.solve_atr["Xd0"]
        timerange = self.solve_atr["time_range"]
        solver_alg = self.solve_atr["solver_alg"]

        solver = solvers.LMM_sys(self.M, self.C, self.K, self.T, X0, Xd0, timerange)

        t = time.time()
        print("Solution started")
        sol = solver.solve_de(solver_alg)
        print(solver_alg, "time: ", time.time() - t)

        return sol

    def get_natural_freqs(self):
        """
        Computes the eigenvalues of the lumped-mass system to validate the model based on models presented in
        literature.

        Returns
        -------
        Natural frequencies and eigenvalues
        """

        K = self.K_b + self.K_e(0)  # + (PG.Omega_c)**2 * PG.K_Omega

        val, vec = sci.linalg.eig(K, self.M)
        indexes = np.argsort(val)

        val = val[indexes]
        eig_freqs = np.sqrt(val) / (np.pi * 2)
        vec = vec[indexes]

        distinct = []
        multiplicity = 1
        for i in range(1, len(val)):
            print(eig_freqs[i])
            if abs(eig_freqs[i - 1] - eig_freqs[i]) < 1:
                multiplicity += 1
            else:
                distinct.append([np.real(eig_freqs[i - 1]), multiplicity])
                multiplicity = 1
        print(np.array(distinct))

    def get_transducer_vibration(self):
        transp = Transmission_Path(self)
        return transp.y()

    def get_windows(self, window_length):
        tp = Transmission_Path(self)
        y = tp.y()
        winds = tp.window_extract(window_length, y, self.Omega_c, self.fs)
        return winds
