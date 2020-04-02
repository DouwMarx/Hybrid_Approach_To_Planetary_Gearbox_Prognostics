
import argparse
import cv2
import numpy as np
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

class Image_Calibration(object):
    """
    Processes the array obtained from the open cv plot though clicks
    """

    def __init__(self, selected_points, cut_depth, cut_breadth):
        """
        Initializes the image calibration object
        Parameters
        ----------
        selected_points: a  6x2 numpy array with coordinates as obtained from clicking on the image
        """


        self.cut_depth = cut_depth
        self.cut_breadth = cut_breadth

        self.points = selected_points

        self.vline = self.vlinecomp()
        self.hline = self.ortho_line_cut()

        self.mid_left = self.midpoint(0,1)
        self.mid_right = self.midpoint(2, 3)

    def gradient(self,i,f):
        """
        Computes the gradient between point i and point f
        Parameters
        ----------
        i: int
            First point index

        f: int
            End point index

        Returns
        -------
        gradient: float
                The computed gradient
        """

        diff = self.points[f, :] - self.points[i, :]
        gradient = diff[1]/diff[0]

        return gradient

    def average_grad(self):
        """
        Computes the average gradient of the respective vertical lines drawn on the sides of the edm cut
        Returns
        -------
        ave_grad: float
                Average gradient of the two lines

        """

        #  Compute the respective gradients
        grad_line_1 = self.gradient(0,1)
        grad_line_2 = self.gradient(2,3)

        a1 = np.abs(np.arctan(grad_line_1))
        a2 = np.abs(np.arctan(grad_line_2))

        ave_grad = np.tan((a1+a2)/2)

        #ave_grad = np.average([grad_line_1,grad_line_2]) # Compute the average gradient

        return ave_grad

    def fitline(self,i,f):
        """
        finds the parameters of a line m and c between the specified points. Thank you Stack overflow: Joran Beasley

        Parameters
        ----------
        i: First point index
        f: Second point index

        Returns
        -------
        m: gradient
        c: y-offset
        """
        #from numpy import ones, vstack
        #from numpy.linalg import lstsq
        points = [(self.points[i,0], self.points[i, 1]), (self.points[f, 0], self.points[f, 1])]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords,rcond=None)[0]
        return m,c

    def intersect_point(self,m1,c1,m2,c2):
        """
        Computes the intersection point of two lines
        Parameters
        ----------
        m1: Grad 1
        c1: y-offset 1
        m2: Grad 2
        c2: y-offset 2

        Returns
        -------
        x,y the computed coordinates of the intersection
        """

        x = (c2 - c1)/(m1 - m2)
        y = m1*x + c1
        return x, y

    def line_through_point(self,grad,i):
        """
        Computes the parameters of a line with a given gradient through a point
        Parameters
        ----------
        grad: gradient
        i: index of point in the objects points attribute

        Returns
        -------
        m,c  The gradient and y-offset of the line
        """

        y = self.points[i,1]
        x = self.points[i, 0]
        m = grad

        c = y - m*x

        return m, c

    def vlinecomp(self):
        """
        Computes the start point, end point and distance of the vertical calibration line
        Returns
        -------
        start: list of coordinate
        end  : list of coordinates
        length: float that shows the length of the line
        """
        m_h, c_h = self.fitline(0,2) # Computes the equation for a line joining the points on the outside of the gear on opposites sides of the edm cut

        m_v_avg = self.average_grad()  # Computes the average gradient of the constructed vertical line

        m_v_avg, c_v = self.line_through_point(m_v_avg,4)  # Equation of line with average gradient though crack start point

        x_intersect,y_intersect = self.intersect_point(m_h, c_h, m_v_avg, c_v)

        coord_top = [x_intersect,y_intersect]
        coord_bot = [self.points[4, 0], self.points[4, 1]]

        distance = self.distance(coord_bot,coord_top)

        return coord_top, coord_bot, distance

    def distance(self,coord_1, coord_2):
        """Takes two lists of coordinates and computes the distance between then"""
        return np.sqrt(np.sum((np.array(coord_1)-np.array(coord_2))**2))

    def midpoint(self,i,f):
        """
        Computes the midpoint between point i and point f
        Parameters
        ----------
        i: int
            First point index

        f: int
            End point index

        Returns
        -------
        gradient: float
                The computed gradient
        """

        summation = self.points[f, :] + self.points[i, :]
        midploint = summation/2
        x_mid = midploint[0]
        y_mid = midploint[1]

        return x_mid,y_mid

    def ortho_line_cut(self):
        """
        Computes the parameters of the line that joins the mid point of the one side to a line parallel to the vertical line that passes through the other side mid point
        Returns
        -------
        start: list of coordinate
        end  : list of coordinates
        length: float that shows the length of the line
        """
        x_mid_left, y_mid_left = self.midpoint(0,1)  # Computes the mid point of the LHS face of the edm cut
        x_mid_right, y_mid_right = self.midpoint(2,3) # Computes the mid point of the RHS face of the edm cut

        ave_grad = self.average_grad()
        m_horizontal = -1/ave_grad  #90 degrees rotation of the vertical line average gradient

        horizontal_eq_c = y_mid_right - m_horizontal*x_mid_right  # y offset of horizontal line
        vertical_eq_left_c = y_mid_left - ave_grad * x_mid_left   # y offset of vertical line on left side

        x_intersect, y_intersect = self.intersect_point(m_horizontal, horizontal_eq_c, ave_grad,vertical_eq_left_c)


        coordleft = [x_intersect, y_intersect]
        coordright =[x_mid_right, y_mid_right]

        dist = self.distance(coordleft, coordright)

        return coordleft, coordright, dist

    def crack_legth_compute(self):
        """
        Computes the length of the crack relative to the references
        Returns
        -------
        """
        crack_grad = self.gradient(4,5)
        avegrad = self.average_grad()
        cracklen = self.distance(self.points[4,:],self.points[5,:])
        #print("photo_crack_len", cracklen)

        #print("Crack_grad:", crack_grad)

        angle_between_crack_and_vertical = np.arctan(np.abs((crack_grad - avegrad)/(1+crack_grad*avegrad)))
        #print("angle between crack and vertical: ", np.rad2deg(angle_between_crack_and_vertical),"degrees")


        h_dist = np.sin(angle_between_crack_and_vertical)*cracklen
        h_rel = h_dist/self.hline[2]
        #print('relative horizontal_crack distance:', h_rel)
        v_dist = np.cos(angle_between_crack_and_vertical)*cracklen
        v_rel = v_dist / self.vline[2]
        #print('relative vertical_crack distance:', v_rel)

        actual_crack_length = np.sqrt((h_rel*self.cut_breadth)**2 + (v_rel*self.cut_depth)**2)

        #print("Actual_crack_length\n,", actual_crack_length)
        print(actual_crack_length)  # This is later writen to a text file and read into the main program

        return actual_crack_length

def line_plot(array_index_start,array_index_finish):
    """
    Takes the pointstore array with a certain position and plots a line between them
    Parameters
    ----------
    array_index_start: int
                        start the line here

    array_index_finish: int
                        end the line here
    Returns
    -------

    """
    cv2.line(image, (int(pointstore[array_index_start, 0]), int(pointstore[array_index_start, 1])), (int(pointstore[array_index_finish, 0]), int(pointstore[array_index_finish, 1])),
             (0, 0, 0), 1)

def line_plot_coord(list_start,list_finish):
    """
    does the same a lineplot but for list inputs
    Parameters
    ----------
    list_start: int
                        start the line here

    list_finish: int
                        end the line here
    Returns
    -------

    """
    cv2.line(image, (int(list_start[0]), int(list_start[1])), (int(list_finish[0]), int(list_finish[1])),
             (255, 255, 255), 1)

    grad = np.array(list_finish) - np.array(list_start)
    grad = grad[1]/grad[0]
    # orthograd = -1/grad
    # r = 5
    # theta = np.abs(np.arctan(grad))
    # delta_y = np.sin(theta+np.pi/2)*r
    # delta_x = np.cos(theta+np.pi/2)*r

    #cv2.line(image, (int(list_start[0]- delta_x) , int(list_start[1]- delta_y)), (int(list_start[0] + delta_x), int(list_start[1] + delta_y)),
     #        (255, 255, 255), 1)

    #cv2.line(image, (int(list_finish[0]- delta_x) , int(list_finish[1]- delta_y)), (int(list_finish[0] + delta_x), int(list_finish[1] + delta_y)),
     #        (255, 255, 255), 1)

def point_select(event, x, y, flags, param):
    # grab references to the global variables
    global points, pointstore, counter

    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        pointstore[counter,:] = [points[0][0],points[0][1]]
        counter+=1
        # check to see if the left mouse button was released

    #Plot the first two dots
    elif event == cv2.EVENT_LBUTTONUP and counter<3:
        cv2.circle(image, points[0], 1, (102, 255, 0), -1)
        cv2.imshow("image", image)


       # if counter ==2:
           # line_plot(0, 1)

    #Plot the second two dots and the first line
    elif event == cv2.EVENT_LBUTTONUP and counter<5:
        cv2.circle(image, points[0], 1, (250, 237, 39), -1)
        cv2.imshow("image", image)

       # if counter ==4:
           # line_plot(2, 3)

    #  Plot crack dots and second line
    elif event == cv2.EVENT_LBUTTONUP and counter<7:
        cv2.circle(image, points[0], 1, (248, 24, 148), -1)
        cv2.imshow("image", image)

        if counter ==6:
            line_plot(4,5)

            obj = Image_Calibration(pointstore,l, w)  #  Notce that the wire machined slot parameters are set here
            v_start, v_fin, v_len = obj.vline
            h_start, h_fin, h_len = obj.hline

            #plot the midpoints
            #cv2.circle(image, (int(obj.mid_left[0]),int(obj.mid_left[1])), 2, (248, 24, 148), -1)
            #cv2.circle(image, (int(obj.mid_right[0]), int(obj.mid_right[1])), 2, (248, 24, 148), -1)

            # Plot the horizontal an vertcal axis systems
            line_plot_coord(h_start, h_fin)
            line_plot_coord(v_start, v_fin)

            length = obj.crack_legth_compute()
            #cv2.imwrite("Testim.png")

            # Write length on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (25, 420)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2

            cv2.putText(image, str(np.round(length,2)) + " mm",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-j", "--w", required=True, help="cut width")
ap.add_argument("-k", "--l", required=True, help="cut length")

args = vars(ap.parse_args())
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])

w = np.float(args["w"])
l = np.float(args["l"])



points = []
pointstore = np.zeros((7,2))
pointstore.astype(int)
counter = 0


#image = cv2.imread("test_im.jpg")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", point_select)
# keep looping until the 'q' key is pressed
while counter< 7:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break



save_name = args["image"][0:-4] + "_mod.png"

cv2.imwrite(save_name, image)

cv2.destroyAllWindows()

