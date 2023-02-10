import numpy as np
import cv2
from numba import jit
from sympy import Line

class CourtDetector:

    """
    Detecting and tracking court in frame

    """
    def __init__(self):

        self.dist_tau = 3
        self.intensity_threshold = 40
        self.v_width = 0
        self.v_height = 0
        self.ymin = (4,4)
        self.ymax = (267,4)
        self.xmin = (4,509)
        self.xmax = (267,509)

     
    def detect_court(self,image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       #define white pixels threshold   
        gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)[1]
       #filter white pixels into court line/not court line 
        filtered = self.filter_pixels(gray)
       #get horizontal and vertical lines merged to get intersection points 
        horizontal, vertical = self.detect_lines(filtered,image)
        points = self.get_points(horizontal,vertical)
        edge = np.squeeze(cv2.convexHull(points))
        # limit contour to quadrilateral
        peri = cv2.arcLength(edge, True)
        corners = np.squeeze(cv2.approxPolyDP(edge, 0.01 * peri, True))
        corners = self.sort_intersection_points(corners)
        Moments = cv2.moments(edge)

        if Moments["m00"] != 0:
            x = int(Moments["m10"] / Moments["m00"])
            y = int(Moments["m01"] / Moments["m00"])

            return (edge,(x, y),points,corners)
 
    def filter_pixels(self,gray):
        
        """
        Filter pixels by using the court line structure
        """
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0:
                    continue
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and
                        gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue

                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and
                        gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                gray[i, j] = 0
        return gray


    def display_lines_on_frame(self,frame, horizontal, vertical):

        for line in horizontal:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #cv2.circle(frame, (x1, y1), 1, (0, 255, 0), 2)
        #cv2.circle(frame, (x2, y2), 1, (0, 255, 0), 2)

        for line in vertical:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #cv2.circle(frame, (x1, y1), 1, (0, 255, 0), 2)
        #cv2.circle(frame, (x2, y2), 1, (0, 255, 0), 2)


    def classify_lines(self,lines):
        """
        Classify line to vertical and horizontal lines
        """
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 2 * dy:
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        # Filter horizontal lines using vertical lines lowest and highest point
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)

        return clean_horizontal, vertical

    def line_intersection(self,line1, line2):

        l1 = Line(line1[0], line1[1])
        l2 = Line(line2[0], line2[1])

        intersection = l1.intersection(l2)

        return intersection[0].coordinates


    def merge_lines(self,horizontal_lines, vertical_lines):

        """
        Merge lines that belongs to the same frame`s lines
        """

        # Merge horizontal lines
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        # Merge vertical lines
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = self.line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = self.line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines

    def get_points(self,horizontal_lines, vertical_lines):

        Px=[]
        Py=[] 
        for lineh in horizontal_lines:
            xh1, yh1, xh2, yh2 = lineh
            for linev in vertical_lines:
                xv1, yv1, xv2, yv2 = linev
                x,y = self.line_intersection(((xh1, yh1), (xh2, yh2)), ((xv1, yv1), (xv2, yv2)))
                x,y = round(eval(str(x))),round(eval(str(y)))
                Px.append(x)
                Py.append(y)

        points = np.float32(np.column_stack((Px,Py)))

        return np.round(points).astype(int)


    def detect_lines(self,gray,frame):
        """
        Finds all line in frame using Hough transform
        """
        minLineLength = 100
        maxLineGap = 20
        self.v_height, self.v_width = frame.shape[:2]
        # Detect all lines
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
        lines = np.squeeze(lines)

        # Classify the lines using their slope
        horizontal, vertical = self.classify_lines(lines)

        # Merge lines that belong to the same line on frame
        horizontal, vertical = self.merge_lines(horizontal, vertical)

        return horizontal, vertical

    def sort_intersection_points(self,intersections):
        """
        sort intersection points from top left to bottom right
        """
        y_sorted = sorted(intersections, key=lambda x: x[1])
        p12 = y_sorted[:2]
        p34 = y_sorted[2:]
        p12 = sorted(p12, key=lambda x: x[0])
        p34 = sorted(p34, key=lambda x: x[0])
        p12 = list(map(list,p12[:]))
        p34 = list(map(list,p34[:]))

        return np.array(p12 + p34)