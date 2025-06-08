image = 'testImage.png'
# Path to your input image

# Thresholds for preprocessing
GAUSSIAN_BLUR_KERNEL = (7, 7)
SHARPEN_KERNEL = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]

# Thresholds for contour filtering
MIN_CONTOUR_AREA = 500

# Harris corner detection parameters
HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_THRESHOLD_RATIO = 0.01

# ApproxPolyDP epsilon range
EPSILON_START = 0.01
EPSILON_END = 0.1
EPSILON_STEPS = 20

# Drawing styles
CONTOUR_COLOR = (0, 255, 0)       # Green
CORNER_COLOR = (0, 0, 255)        # Red
SIDE_LINE_COLOR = (255, 0, 0)     # Blue
CORNER_RADIUS = 4
CORNER_THICKNESS = -1
SIDE_LINE_THICKNESS = 2
CONTOUR_THICKNESS = 3


