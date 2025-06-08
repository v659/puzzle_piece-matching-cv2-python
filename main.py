import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import constants



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

    def find_pieces(self, min_area: int = 500):
        inverted = cv2.bitwise_not(self.binary_image)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug:
            print(f"[INFO] Found {len(contours)} contours total.")

        pieces = []
        bounding_boxes = []
        contour_image = self.original_image.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get 4 corner points using rotated rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            if self.debug:
                print(f"[DEBUG] Rectangle corner points: {box.tolist()}")

            # Draw lines between the 4 corners
            for i in range(4):
                pt1 = tuple(box[i])
                pt2 = tuple(box[(i + 1) % 4])
                cv2.line(contour_image, pt1, pt2, (255, 0, 0), 2)  # Blue lines

                # Optional: Draw corner points
                cv2.circle(contour_image, pt1, 5, (0, 255, 255), -1)  # Yellow dots

            # Crop the piece
            x, y, w, h = cv2.boundingRect(contour)
            piece_crop = self.original_image[y:y + h, x:x + w].copy()
            pieces.append(piece_crop)
            bounding_boxes.append((x, y, w, h))

        # Harris corner detection
        harris = cv2.cornerHarris(self.binary_image, 2, 3, 0.04)
        harris = cv2.dilate(harris, None)


        # Draw detected corners on binary image
        corner_overlay = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
        threshold = 0.01 * harris.max()
        corner_points = np.argwhere(harris > threshold)
        for y, x in corner_points:
            cv2.circle(corner_overlay, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

        if self.debug:
            print(f"[INFO] Extracted {len(pieces)} puzzle piece(s). Harris max: {harris.max()}, threshold: {threshold}")
            print(bounding_boxes)

        return pieces, bounding_boxes, contour_image, corner_overlay


def main():
    converter = ManipulateImage(constants.image)
    result = converter.preprocess_image()

    piece_detector = PuzzlePieceDetector(result.binary, result.original, debug=True)
    pieces, boxes, contour_vis, corner_vis = piece_detector.find_pieces(min_area=500)

    print(f"[INFO] Extracted {len(pieces)} puzzle piece(s)")

    show_image([
        result.original,
        result.gray,
        result.sharpened,
        result.blurred,
        result.binary,
        contour_vis,
        corner_vis
    ], title="Stage")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
