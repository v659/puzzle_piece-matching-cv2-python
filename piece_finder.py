import cv2
import numpy as np

def detect_side_bulge(side_mask):
    contours, _ = cv2.findContours(side_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) < 3:
            return "flat"
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None and len(defects) > 0:
            return "inward"  # concave
        else:
            return "outward"  # convex
    return "flat"

# === Load image ===
img = cv2.imread("ChatGPT Image Jun 5, 2025, 03_14_50 PM.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Binarize (invert so pieces are white)
_, binary = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)

# Find contours of puzzle pieces
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Harris corner detection parameters
blockSize = 2
ksize = 3
k = 0.04
dst = cv2.cornerHarris(binary, blockSize, ksize, k)
dst = cv2.dilate(dst, None)
threshold = 0.01 * dst.max()
corners = np.argwhere(dst > threshold)  # (y,x)
corners_xy = np.array([(x, y) for y, x in corners], dtype=np.int32)

# Convert binary to BGR to draw colored overlays
binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# Draw contours and bounding boxes, label pieces


# For each piece: find corners inside, draw hull, analyze and label sides
side_thickness = 10  # thickness for side region

for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) < 1000:
        continue

    # Filter corners inside this piece
    corners_in_piece = []
    for (x_c, y_c) in corners_xy:
        if cv2.pointPolygonTest(cnt, (int(x_c), int(y_c)), False) > 0:
            corners_in_piece.append([x_c, y_c])

    corners_in_piece = np.array(corners_in_piece, dtype=np.int32)
    if len(corners_in_piece) == 0:
        print(f"Piece {i+1}: No corners inside.")
        continue

    # Draw hull around corners
    corners_contour = corners_in_piece.reshape((-1, 1, 2))

    x, y, w, h = cv2.boundingRect(cnt)
    piece_gray = gray[y:y + h, x:x + w]

    # Use Canny edge detection
    edged = cv2.Canny(binary, 30, 200)

    # Find contours in Canny edges
    edge_contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Offset and draw contours on binary_color
    for ec in edge_contours:
        ec += np.array([[x, y]])
        cv2.drawContours(binary_color, [ec], -1, (255, 0, 0), 1)

    # Draw corner points in yellow
    for (x_c, y_c) in corners_in_piece:
        cv2.circle(binary_color, (x_c, y_c), 3, (0, 255, 255), -1)

    # Extract bounding box for the piece to get side slices
    x, y, w, h = cv2.boundingRect(cnt)

    # Create mask for the piece itself
    piece_mask = np.zeros_like(binary)
    cv2.drawContours(piece_mask, [cnt], -1, 255, -1)
    piece_roi = piece_mask[y:y + h, x:x + w]

    # Extract 4 side regions of the piece mask
    side_regions = {
        "top":    piece_roi[0:side_thickness, :],
        "bottom": piece_roi[-side_thickness:, :],
        "left":   piece_roi[:, 0:side_thickness],
        "right":  piece_roi[:, -side_thickness:]
    }

    # Analyze each side using detect_side_bulge and label
    for side, region in side_regions.items():
        shape = detect_side_bulge(region)
        # Compute label position
        tx, ty = x, y
        if side == "top":
            tx = x + w // 4
            ty = y - 15
        elif side == "bottom":
            tx = x + w // 4
            ty = y + h + 20
        elif side == "left":
            tx = x - 60
            ty = y + h // 2
        elif side == "right":
            tx = x + w + 5
            ty = y + h // 2

        cv2.putText(binary_color, f"{side}: {shape}", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"Piece {i+1}: Drew hull around {len(corners_in_piece)} corners and labeled sides.")

import matplotlib.pyplot as plt

plt.imshow(cv2.cvtColor(binary_color, cv2.COLOR_BGR2RGB))
plt.title("Puzzle Pieces with Accurate Contours and Side Labels")
plt.axis('off')
plt.show()

