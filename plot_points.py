import matplotlib.pyplot as plt
import numpy as np

# Third set of points
points3 = [
    (np.int64(65), np.int64(301)), (np.int64(317), np.int64(304)), (np.int64(78), np.int64(304)),
    (np.int64(154), np.int64(305)), (np.int64(230), np.int64(305)), (np.int64(244), np.int64(305)),
    (np.int64(257), np.int64(305)), (np.int64(306), np.int64(306)), (np.int64(161), np.int64(312))
]

# Convert to numpy array
pts3 = np.array(points3)
x_vals = pts3[:, 0]
y_vals = pts3[:, 1]

# Set up plot
fig, ax = plt.subplots(figsize=(6, 6))  # square canvas

# Plot points
ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='blue')

# Set equal intervals (aspect ratio 1:1)
ax.set_aspect('equal', adjustable='datalim')

# Optional: invert Y for image-like coordinates
ax.invert_yaxis()

# Titles and labels
ax.set_title("Third Set of Points (Equal Intervals & Square Axes)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)

plt.show()
