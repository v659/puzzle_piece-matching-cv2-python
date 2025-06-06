import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import constants


def show_image(image, title='Image'):
    if isinstance(image, list):
        for idx, img in enumerate(image):
            cv2.imshow(f"{title} {idx}", img)
    else:
        cv2.imshow(title, image)


class ManipulateImage():
    def __init__(self, image):
        self.image = cv2.imread(image)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image}")

    def threshold_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        # Moderate Gaussian blur for noise smoothing
        blurred = cv2.GaussianBlur(sharpened, (7, 7), 0)
        # Otsu's threshold (global), inverse binary
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological opening with a rectangular kernel to remove thin lines (like grids) but preserve shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Optional: Morphological closing to fill small holes
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        return cleaned, self.image, gray, sharpened, blurred, binary, opened


class GetPieceContours():
    def __init__(self, clean_image):
        self.clean_image = clean_image

    def find_contours(self):

        edged = cv2.Canny(self.clean_image, 30, 200)
        
        return edged






def main():
    converter = ManipulateImage(constants.image)
    contour_finder = GetPieceContours(converter.threshold_image()[0])
    show_image(contour_finder.clean_image, 'Clean Image')
    show_image(contour_finder.find_contours(), 'contours')
    cv2.waitKey(0)
    cv2.destroyAllWindows()



main()
