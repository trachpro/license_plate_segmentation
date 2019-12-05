import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

class LicensePlate():
    def __init__(self):
        pass

    def find_plate(self, image, image_debug_dir=None):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, w, _ = image.shape
        cut_image = image[int(h/3): int(2 * h / 3), int(w/3): int(2 * w / 3), :]
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        dilated_image = cv2.dilate(cut_image, (7, 7), iterations=2)
        cv2.imshow('image', image)
        cv2.imshow('cut_image', cut_image)
        lower, upper = self.get_threshold(dilated_image)
        mask = cv2.inRange(image, lower, upper)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

    def get_threshold(self, image):
        r = cv2.calcHist([image], [0], None, [256], [0, 256])
        g = cv2.calcHist([image], [1], None, [256], [0, 256])
        b = cv2.calcHist([image], [2], None, [256], [0, 256])
        max_r = np.argmax(r)
        max_g = np.argmax(g)
        max_b = np.argmax(b)
        print()
        color_range = 10
        return np.array([max_r-color_range, max_g-color_range, max_b-color_range]), \
               np.array([max_r + color_range, max_g + color_range, max_b + color_range])

if __name__ == '__main__':
    license_path = 'lisence_image'
    license_detector = LicensePlate()
    for path in os.listdir(license_path):
        image = cv2.imread(os.path.join(license_path, path))
        license_detector.find_plate(image)

    cv2.destroyAllWindows()