import cv2
import numpy as np
import matplotlib.pyplot as plt


class CourtReference:
    """
    Court reference model
    """
    def __init__(self):
        self.baseline_top = ((2, 2), (267, 2))
        self.baseline_top_inner = ((2, 37), (267, 37))
        self.baseline_bottom = ((2, 509), (267, 509))
        self.baseline_bottom_inner = ((2, 474), (267, 474))
        self.net = ((2, 255), (267, 255))
        self.left_court_line = ((2, 2), (2, 509))
        self.right_court_line = ((267, 2), (267, 509))
        self.left_inner_line = ((25, 2), (25, 509))
        self.right_inner_line = ((244, 2), (244, 509))
        self.middle_top = ((134, 2), (134, 184))
        self.middle_bottom = ((134, 327), (134, 509))
        self.top_inner_line = ((2, 184), (267, 184))
        self.bottom_inner_line = ((2, 327), (267, 327))

        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
                               self.right_inner_line[1]],
                           3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
                               self.right_court_line[1]],
                           4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
                               self.right_inner_line[1]],
                           5: [(25, 184), (244, 184), (25, 327), (244, 327)],
                           6: [(25, 184), (244, 184), self.left_inner_line[1], self.right_inner_line[1]],
                           7: [self.left_inner_line[0], self.right_inner_line[0], (25, 327), (244, 327)],
                           8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
                               self.right_court_line[1]],
                           9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[0],
                               self.left_inner_line[1]],
                           10: [self.left_inner_line[0], self.right_inner_line[0], (25, 184), (244, 184)],
                           11: [self.left_court_line[0], self.left_inner_line[0], self.baseline_top_inner[0],
                                (25, 37)],
                           12: [(25, 327), (244, 327), self.left_inner_line[1], self.right_inner_line[1]],
                           13: [self.right_inner_line[0], self.right_court_line[0], (244, 37), self.baseline_top_inner[1]],
                           14: [self.baseline_bottom_inner[0], (25, 474), self.baseline_bottom[0], self.left_inner_line[1]],
                           15: [(244, 474), self.baseline_bottom_inner[1], self.right_inner_line[1], self.right_court_line[1]],
                           16: [self.baseline_top[0], self.middle_top[0], self.top_inner_line[0], self.middle_top[1]],
                           17: [self.middle_top[0], self.right_court_line[0], self.middle_top[1], self.top_inner_line[1]],
                           18: [self.bottom_inner_line[0], self.middle_bottom[0], self.baseline_bottom[0], self.middle_bottom[1]],
                           19: [self.middle_bottom[0], self.bottom_inner_line[1], self.middle_bottom[1], self.baseline_bottom[1]],
                           20: [self.bottom_inner_line[0], self.bottom_inner_line[1], self.baseline_bottom[0], self.baseline_bottom[1]],
                           21: [self.baseline_top[0], self.baseline_top[1], self.top_inner_line[0], self.top_inner_line[1]],
                           22: [self.left_court_line[0], self.left_inner_line[0], self.top_inner_line[0], (25, 184)],
                           23: [self.left_court_line[0], self.left_inner_line[0], self.bottom_inner_line[0], (25, 327)],}


        self.line_width = 1


        self.court = cv2.cvtColor(cv2.imread('D:/EFREI/project/Badminton_Project/new court/court.jpg'), cv2.COLOR_BGR2GRAY)

    def build_court_reference(self):
        """
        Create court reference image using the lines positions
        """
        court = np.zeros((512, 270), dtype=np.uint8)
        cv2.line(court, *self.baseline_top, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom, 1, self.line_width)
        cv2.line(court, *self.baseline_top_inner, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom_inner, 1, self.line_width)
        cv2.line(court, *self.top_inner_line, 1, self.line_width)
        cv2.line(court, *self.bottom_inner_line, 1, self.line_width)
        cv2.line(court, *self.left_court_line, 1, self.line_width)
        cv2.line(court, *self.right_court_line, 1, self.line_width)
        cv2.line(court, *self.left_inner_line, 1, self.line_width)
        cv2.line(court, *self.right_inner_line, 1, self.line_width)
        cv2.line(court, *self.middle_top, 1, self.line_width)
        cv2.line(court, *self.middle_bottom, 1, self.line_width)
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))
        plt.imsave('D:/EFREI/project/Badminton_Project/new court/court1.jpg', court, cmap='gray')
        self.court = court
        return court

    def get_important_lines(self):
        """
        Returns all lines of the court
        """
        lines = [*self.baseline_top, *self.baseline_bottom, *self.baseline_top_inner, *self.baseline_bottom_inner, *self.net,
                 *self.left_court_line, *self.right_court_line,
                 *self.left_inner_line, *self.right_inner_line, *self.middle_top, *self.middle_bottom,
                 *self.top_inner_line, *self.bottom_inner_line]
        return lines

    def get_extra_parts(self):
        parts = [self.top_extra_part, self.bottom_extra_part]
        return parts

    def save_all_court_configurations(self):
        """
        Create all configurations of 4 points on court reference
        """
        for i, conf in self.court_conf.items():
            c = cv2.cvtColor(255 - self.court, cv2.COLOR_GRAY2BGR)
            for p in conf:
                c = cv2.circle(c, p, 15, (0, 0, 255), 30)
            cv2.imwrite(f'D:/EFREI/project/Badminton_Project/new court/court_{i}.png', c)

    def get_court_mask(self, mask_type=0):
        """
        Get mask of the court
        """
        mask = np.ones_like(self.court)
        if mask_type == 1:  # Bottom half court
            mask[:self.net[0][1] - 1000, :] = 0
        elif mask_type == 2:  # Top half court
            mask[self.net[0][1]:, :] = 0
        elif mask_type == 3: # court without margins
            mask[:self.baseline_top[0][1], :] = 0
            mask[self.baseline_bottom[0][1]:, :] = 0
            mask[:, :self.left_court_line[0][0]] = 0
            mask[:, self.right_court_line[0][0]:] = 0
        return mask


if __name__ == '__main__':
    c = CourtReference()
    #c.build_court_reference()