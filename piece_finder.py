import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
import constants


def show_image(images, title='Image'):
    if not isinstance(images, list):
        images = [images]

    # Display all images in a grid using matplotlib
    cols = 4
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:  # Grayscale
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()



def _display(image, title):
    if os.environ.get("DISPLAY", "") == "":
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        cv2.imshow(title, image)


@dataclass
class PreprocessingResult:
    cleaned: np.ndarray
    original: np.ndarray
    gray: np.ndarray
    sharpened: np.ndarray
    blurred: np.ndarray
    binary: np.ndarray
    opened: np.ndarray


class ManipulateImage:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

    def preprocess_image(self) -> PreprocessingResult:
        # Convert to grayscale
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

        # Morphology to remove gridlines and noise
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel, iterations=2)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

        return PreprocessingResult(
            cleaned=cleaned,
            original=self.image,
            gray=gray,
            sharpened=sharpened,
            blurred=blurred,
            binary=binary,
            opened=opened
        )


class PieceContourFinder:
    def __init__(self, image):
        self.image = image

    def find_edges(self):
        return cv2.Canny(self.image, 30, 200)


def main():
    converter = ManipulateImage(constants.image)
    result = converter.preprocess_image()

    # Edge detection
    contour_finder = PieceContourFinder(result.binary)
    edges = contour_finder.find_edges()

    # Show all stages
    show_image([
        result.original,
        result.gray,
        result.sharpened,
        result.blurred,
        result.binary,
        result.opened,
        result.cleaned,
        edges
    ], title='Stage')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
