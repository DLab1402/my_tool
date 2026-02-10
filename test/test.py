import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# sample data
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 0, 1, 0])

# create spline
cs = CubicSpline(x, y)

# smooth x values
x_smooth = np.linspace(0, 4, 200)
y_smooth = cs(x_smooth)

# plot
plt.plot(x, y, 'o')          # original points
plt.plot(x_smooth, y_smooth) # spline curve
plt.show()