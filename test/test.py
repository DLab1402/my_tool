import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Sample data
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 0, 1, 0])

# Create spline
cs = CubicSpline(x, y)

# Evaluate on finer grid
x_fine = np.linspace(0, 4, 200)
y_fine = cs(x_fine)

plt.plot(x, y, 'o', label="data")
plt.plot(x_fine, y_fine, label="cubic spline")
plt.legend()
plt.show()
