import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random

def show_image(images, title='Image', side_points_list=None):
    if not isinstance(images, list):
        images = [images]
    if side_points_list is None:
        side_points_list = [None] * len(images)

    cols = 4
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))

    for i, (img, side_points) in enumerate(zip(images, side_points_list)):
        vis = img.copy()

        if side_points:
            for group in side_points:
                # BLUE CORNERS (4-point corner list)
                if isinstance(group, list) and len(group) == 4 and all(isinstance(pt, tuple) and len(pt) == 2 for pt in group):
                    for pt in group:
                        x, y = map(int, pt)
                        cv2.circle(vis, (x, y), radius=6, color=(255, 0, 0), thickness=-1)  # Blue
                    continue

                # SINGLE POINT
                if isinstance(group, tuple) and len(group) == 2:
                    cv2.circle(vis, (int(group[0]), int(group[1])), radius=2, color=(0, 0, 255), thickness=-1)

                # FLAT HARRIS POINT LIST
                elif isinstance(group, list) and all(isinstance(pt, tuple) and len(pt) == 2 for pt in group):
                    for pt in group:
                        x, y = map(int, pt)
                        cv2.circle(vis, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

                # SIDE POINT GROUP
                elif isinstance(group, list):
                    color = tuple(random.randint(0, 255) for _ in range(3))
                    for pt in group:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), radius=2, color=color, thickness=-1)
                    for j in range(len(group) - 1):
                        cv2.line(vis,
                                 (int(group[j][0]), int(group[j][1])),
                                 (int(group[j + 1][0]), int(group[j + 1][1])),
                                 color, 1)

        plt.subplot(rows, cols, i + 1)
        if len(vis.shape) == 2:
            plt.imshow(vis, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
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


from sklearn.cluster import DBSCAN

class PuzzlePieceDetector:
    def __init__(self, binary_image: np.ndarray, original_image: np.ndarray, debug: bool = False):
        self.binary_image = binary_image
        self.original_image = original_image
        self.debug = debug

    def deduplicate_points(self, points, radius=5):
        if not points:
            return []
        X = np.array(points)
        clustering = DBSCAN(eps=radius, min_samples=1).fit(X)
        labels = clustering.labels_

        deduped = []
        for label in np.unique(labels):
            cluster_points = X[labels == label]
            centroid = cluster_points.mean(axis=0).astype(int)
            deduped.append((centroid[0], centroid[1]))
        return deduped

    def find_pieces(self, min_area: int = 500):
        inverted = cv2.bitwise_not(self.binary_image)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pieces = []
        bounding_boxes = []
        harris_points_all = []
        contour_image = self.original_image.copy()
        corner_overlay = self.original_image.copy()

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            piece_crop = self.original_image[y:y + h, x:x + w]
            binary_crop = self.binary_image[y:y + h, x:x + w]

            # Harris corners (raw)
            gray = np.float32(binary_crop)
            harris = cv2.cornerHarris(gray, 2, 3, 0.04)
            harris = cv2.dilate(harris, None)
            threshold = 0.01 * harris.max()
            corner_points = np.argwhere(harris > threshold)

            # Convert to (x, y) relative to full image
            corner_points_xy = [(int(pt[1] + x), int(pt[0] + y)) for pt in corner_points]

            # Deduplicate close Harris corners
            deduped_points = self.deduplicate_points(corner_points_xy, radius=3)

            # Draw deduplicated corners
            for px, py in deduped_points:
                cv2.circle(corner_overlay, (px, py), 2, (0, 0, 255), -1)

            if self.debug:
                print(f"[DEBUG] Piece {idx}: {len(corner_points_xy)} raw → {len(deduped_points)} deduped Harris corners.")

            pieces.append(piece_crop)
            bounding_boxes.append((x, y, w, h))
            harris_points_all.append(deduped_points)
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.debug:
            print(f"[INFO] Extracted {len(pieces)} puzzle piece(s).")

        return pieces, bounding_boxes, contour_image, corner_overlay, harris_points_all


    def draw_piece_sides(image, bounding_boxes, side_points_all, colors=None):

        if colors is None:
            # Generate 10 distinct BGR colors (you can add more if needed)
            colors = [
                (255, 0, 0),  # Blue
                (0, 255, 0),  # Green
                (0, 0, 255),  # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 128),  # Purple
                (0, 128, 255),  # Orange
                (128, 255, 0),  # Lime
                (255, 128, 0)  # Light orange
            ]

        output = image.copy()

        for idx, (bbox, sides) in enumerate(zip(bounding_boxes, side_points_all)):
            for i, side in enumerate(sides):
                color = colors[i % len(colors)]
                for j in range(len(side) - 1):
                    pt1 = side[j]
                    pt2 = side[j + 1]
                    cv2.line(output, pt1, pt2, color, 2)

                # Optionally, draw a dot at the start of each side
                if len(side) > 0:
                    cv2.circle(output, side[0], 4, color, -1)

            # Label the piece
            x, y, w, h = bbox
            cv2.putText(output, f'#{idx}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return output


def draw_harris_corners(image, harris_points):
    overlay = image.copy()
    for pt in harris_points:
        cv2.circle(overlay, pt, radius=2, color=(0, 0, 255), thickness=-1)
    return cv2.addWeighted(image, 0.4, overlay, 0.6, 0)


from scipy.spatial.distance import directed_hausdorff
from itertools import combinations

def detect_corners(points, std_tol=0.25, length_bonus_weight=2.0, deviation_penalty_weight=1.0):

    if len(points) < 10:
        return []

    pts = np.array(points)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()

    best_score = float('inf')
    best_combo = None

    for combo in combinations(pts, 4):
        combo = sorted(combo, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        dists = [np.linalg.norm(np.array(combo[i]) - np.array(combo[(i + 1) % 4])) for i in range(4)]

        mean_dist = np.mean(dists)
        std_dev = np.std(dists)

        if std_dev / mean_dist > std_tol:
            continue

        score = -length_bonus_weight * mean_dist + deviation_penalty_weight * std_dev

        if score < best_score:
            best_score = score
            best_combo = combo
    return best_combo if best_combo is not None else []


def extract_sides_from_corners(harris_points, corners):
    print(corners)
    if len(corners) != 4:
        return []

    pts = np.array(harris_points)
    sides = []

    for i in range(4):
        c1 = np.array(corners[i])
        c2 = np.array(corners[(i+1) % 4])
        vec = c2 - c1
        norm = np.linalg.norm(vec) + 1e-5
        direction = vec / norm

        side = []
        for pt in pts:
            rel = pt - c1
            proj = np.dot(rel, direction)

            # Check if point lies between the two corners (in projected direction)
            if 0 <= proj <= norm:
                side.append(tuple(pt))  # Keep all points in side span, including bulges

        sides.append(side)

    return sides



def compare_sides_fuzzy(side1, side2):
    if not side1 or not side2:
        return float('inf')
    a = np.array(side1)
    b = np.array(side2[::-1])  # reversed
    fwd = directed_hausdorff(a, b)[0]
    bwd = directed_hausdorff(b, a)[0]
    return max(fwd, bwd)


def classify_side_shape(side, side_index):
    if len(side) < 2:
        return 'unknown'

    pts = np.array(side)
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]

    # Determine main axis
    dx = np.ptp(x_vals)
    dy = np.ptp(y_vals)

    if dx > dy:
        base_vals = y_vals  # horizontal side
    else:
        base_vals = x_vals  # vertical side

    profile = base_vals.astype(float)
    mean_val = np.mean(profile)
    deviations = profile - mean_val
    abs_devs = np.abs(deviations)
    max_dev = np.max(abs_devs)

    # Use max_dev to determine direction
    furthest_idx = np.argmax(abs_devs)
    direction = deviations[furthest_idx]  # signed deviation from mean
    print(direction)
    # Heuristic by side position
    if side_index in [0, 1]:  # top or right
        shape_type = 'outward' if direction > 0 else 'inward'
    elif side_index in [2, 3]:  # bottom or left
        shape_type = 'outward' if direction < 0 else 'inward'
    else:
        shape_type = 'unknown'

    return shape_type







def match_all_sides(all_sides):
    """Returns list of (piece_a, side_a, piece_b, side_b, score)"""
    matches = []
    for i, sides_i in enumerate(all_sides):
        for j, sides_j in enumerate(all_sides):
            if i >= j:
                continue
            for idx_a, side_a in enumerate(sides_i):
                for idx_b, side_b in enumerate(sides_j):
                    type_a = classify_side_shape(side_a, idx_a)
                    type_b = classify_side_shape(side_b, idx_b)

                    # Only match complementary types
                    if (type_a == 'inward' and type_b == 'outward') or \
                       (type_a == 'outward' and type_b == 'inward'):

                        score = compare_sides_fuzzy(side_a, side_b)
                        matches.append((i, idx_a, j, idx_b, score))
    return sorted(matches, key=lambda x: x[4])



def main():
    converter = ManipulateImage('testImage.png')
    result = converter.preprocess_image()

    piece_detector = PuzzlePieceDetector(result.binary, result.original, debug=True)
    pieces, boxes, contour_vis, corner_vis, harris_points_all = piece_detector.find_pieces(min_area=500)

    # Convert global Harris points to local (relative to piece crop) coordinates
    local_harris_points_all = []
    for (x, y, w, h), points in zip(boxes, harris_points_all):
        local_pts = [(px - x, py - y) for (px, py) in points]
        local_harris_points_all.append(local_pts)

    print(f"[INFO] Extracted {len(pieces)} puzzle piece(s)")

    # === CORNER DETECTION + SIDE MATCHING ===
    print("\n=== Starting Side Detection ===")
    all_sides = []
    corner_groups = []

    for i, harris_pts in enumerate(local_harris_points_all):
        corners = detect_corners(harris_pts)
        corner_groups.append(corners)

        if len(corners) != 4:
            print(f"[WARN] Piece {i} has {len(corners)} corners — skipping side extraction.")
            all_sides.append([])
            continue

        sides = extract_sides_from_corners(harris_pts, corners)
        all_sides.append(sides)
        print(f"[DEBUG] Piece {i} → {len(corners)} corners, {len(sides)} sides.")

    # Match sides
    side_matches = match_all_sides(all_sides)

    print("\n=== Top Side Matches ===")
    for a, sa, b, sb, score in side_matches[:10]:
        print(f"Piece {a} Side {sa} ↔ Piece {b} Side {sb} → Match Score: {score:.2f}")

    for i, sides in enumerate(all_sides):
        print(f"\n🧩 Puzzle Piece {i}")
        for j, side in enumerate(sides):
            shape = classify_side_shape(side, j)
            symbol = {
                'outward': '↪️ Tab (Outward)',
                'inward': '↩️ Slot (Inward)',
                'flat': '▬ Flat Edge',
            }.get(shape, '❓ Unknown')
            print(f"  Side {j}: {symbol}")

    """def reconstruct_and_show_puzzle(pieces, boxes, all_sides, side_matches, canvas_size=(2000, 2000)):
        canvas = np.ones((*canvas_size, 3), dtype=np.uint8) * 255
        placed = {}
        placed_sides = set()

        def place_piece(idx, x, y, angle=0):
            if idx in placed:
                return
            piece = pieces[idx]
            rotated = rotate_image(piece, angle)
            h, w = rotated.shape[:2]
            canvas[y:y + h, x:x + w] = rotated
            placed[idx] = (x, y, angle)

        def rotate_image(image, angle):
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

        place_piece(0, 100, 100, angle=0)  # Place first piece arbitrarily

        for (i, side_i, j, side_j, score) in side_matches:
            if i in placed and j in placed:
                continue
            if (i, side_i) in placed_sides or (j, side_j) in placed_sides:
                continue
            if score > 20.0:  # skip bad matches
                continue

            if i in placed:
                base_idx, base_side = i, side_i
                other_idx, other_side = j, side_j
            elif j in placed:
                base_idx, base_side = j, side_j
                other_idx, other_side = i, side_i
            else:
                continue  # neither placed yet

            base_pts = all_sides[base_idx][base_side]
            other_pts = all_sides[other_idx][other_side]

            if not base_pts or not other_pts:
                continue

            # Compute vector from other to base
            base_vec = np.array(base_pts[-1]) - np.array(base_pts[0])
            other_vec = np.array(other_pts[0]) - np.array(other_pts[-1])  # reversed

            angle1 = np.degrees(np.arctan2(base_vec[1], base_vec[0]))
            angle2 = np.degrees(np.arctan2(other_vec[1], other_vec[0]))
            rotation_angle = angle1 - angle2

            # Get new rotated piece
            rotated_piece = rotate_image(pieces[other_idx], rotation_angle)
            h, w = rotated_piece.shape[:2]

            # Translate other_pts[0] to base_pts[-1] after rotation
            R = cv2.getRotationMatrix2D((pieces[other_idx].shape[1] // 2, pieces[other_idx].shape[0] // 2),
                                        rotation_angle, 1)
            pt = np.dot(R, np.array([other_pts[0][0], other_pts[0][1], 1]))
            dx, dy = np.array(base_pts[-1]) - pt
            x_new = int(placed[base_idx][0] + dx)
            y_new = int(placed[base_idx][1] + dy)

            # Place piece
            canvas[y_new:y_new + h, x_new:x_new + w] = rotated_piece
            placed[other_idx] = (x_new, y_new, rotation_angle)
            placed_sides.add((base_idx, base_side))
            placed_sides.add((other_idx, other_side))

        cv2.imshow("Reconstructed Puzzle", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows() """
    # === VISUALIZATION: Detected Corners as BLUE ===
    side_points_list_with_corners = []
    for harris, corners in zip(local_harris_points_all, corner_groups):
        combined = [harris]
        if corners:
            combined.append(corners)  # triggers blue large-dot drawing
        side_points_list_with_corners.append(combined)
        base_vis = [result.original, result.gray, result.sharpened, result.blurred, result.binary, contour_vis,
                    corner_vis]
        show_image(
            images=base_vis + pieces,
            title="Stage",
            side_points_list=[None] * len(base_vis) + side_points_list_with_corners
        )


if __name__ == "__main__":
    main()

