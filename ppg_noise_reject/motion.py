import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0      # Force magnitude (N)
m = 1.0      # Mass (kg)
w = 2 * np.pi  # Angular frequency (rad/s)

# Time vector
t = np.linspace(0, 100, 1000)

# Position equations
x = (T / (m * w)) * t - (T / (m * w**2)) * np.sin(w * t)
y = (T / (m * w**2)) * (1 - np.cos(w * t))

# Plot trajectory
plt.figure(figsize=(8, 6))
plt.plot(x, y, linewidth=2)
plt.scatter(x[0], y[0], label='Start (t=0)')
plt.scatter(x[-1], y[-1], label='End (t=1)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectory of Particle Under Rotating Force')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()