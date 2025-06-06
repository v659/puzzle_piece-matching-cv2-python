import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import constants


def show_image(image, title='Image'):
    if isinstance(image, list):
        for i, img in enumerate(image):
            cv2.imshow(f"{title} {i}", img)
    else:
        cv2.imshow(title, image)


class ManipulateImage:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

    def threshold_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Sharpen
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

        # Blur
        blurred = cv2.GaussianBlur(sharpened, (7, 7), 0)

        # Threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphology
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel, iterations=2)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

        return cleaned, self.image, gray, sharpened, blurred, binary, opened


class PieceContourFinder:
    def __init__(self, image):
        self.image = image

    def find_edges(self):
        return cv2.Canny(self.image, 30, 200)


def main():
    converter = ManipulateImage(constants.image)
    cleaned, original, gray, sharpened, blurred, binary, opened = converter.threshold_image()

    # Edge detection
    contour_finder = PieceContourFinder(cleaned)
    edges = contour_finder.find_edges()

    # Display results
    show_image([
        original, gray, sharpened, blurred, binary, opened, cleaned, edges
    ], title='Stage')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
