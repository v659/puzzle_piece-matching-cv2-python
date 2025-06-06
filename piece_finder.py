import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import constants
from typing import List, Tuple


def show_image(images, title='Image'):
    if not isinstance(images, list):
        images = list[images]

    cols = 4
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


@dataclass
class PreprocessingResult:

    original: np.ndarray
    gray: np.ndarray
    sharpened: np.ndarray
    blurred: np.ndarray
    binary: np.ndarray


class ManipulateImage:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

    def preprocess_image(self) -> PreprocessingResult:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        blurred = cv2.GaussianBlur(sharpened, (7, 7), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return PreprocessingResult(
            original=self.image,
            gray=gray,
            sharpened=sharpened,
            blurred=blurred,
            binary=binary,

        )


class PuzzlePieceDetector:
    def __init__(self, binary_image: np.ndarray, original_image: np.ndarray, debug: bool = False):
        self.binary_image = binary_image
        self.original_image = original_image
        self.debug = debug

    def find_pieces(self, min_area: int = 500) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray]:
        inverted = cv2.bitwise_not(self.binary_image)
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug:
            print(f"[INFO] Found {len(contours)} contours total.")

        pieces = []
        bounding_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            piece_crop = self.original_image[y:y + h, x:x + w].copy()
            pieces.append(piece_crop)
            bounding_boxes.append((x, y, w, h))
        contour_image = self.original_image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

        if self.debug:
            print(f"[INFO] Extracted {len(pieces)} puzzle piece(s).")

        return pieces, bounding_boxes, contour_image


def main():
    converter = ManipulateImage(constants.image)
    result = converter.preprocess_image()

    piece_detector = PuzzlePieceDetector(result.binary, result.original)
    pieces, boxes, contour_vis = piece_detector.find_pieces(min_area=500)

    print(f"[INFO] Extracted {len(pieces)} puzzle piece(s)")

    show_image([
        result.original,
        result.gray,
        result.sharpened,
        result.blurred,
        result.binary,

    ], title='Stage')

    if pieces:
        show_image(contour_vis, title="Puzzle Piece(s)")
    else:
        print("[WARN] No pieces found.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
