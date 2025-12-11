import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# Parameters and ODE system
# -----------------------------
alpha = 0.01
r0, f0 = 15.0, 22.0      # new initial conditions
y0 = [r0, f0]

def rabbits_foxes(t, y):
    r, f = y
    drdt = 2.0 * r - alpha * r * f
    dfdt = -1.0 * f + alpha * r * f
    return [drdt, dfdt]

# -----------------------------
# Solve with 3rd-order RK (RK23)
# -----------------------------
t0, t_end = 0.0, 40.0              # ~6 periods (period ~ 6.6)
t_eval = np.linspace(t0, t_end, 4001)

sol = solve_ivp(
    rabbits_foxes,
    (t0, t_end),
    y0,
    t_eval=t_eval,
    method="RK23"
)

t = sol.t
r = sol.y[0]
f = sol.y[1]

print(f"Rabbit min ≈ {r.min():.2f}, max ≈ {r.max():.2f}")
print(f"Fox   min ≈ {f.min():.2f}, max ≈ {f.max():.2f}")

# -----------------------------
# Plot 1: r(t) and f(t) vs time
# -----------------------------
plt.figure(figsize=(8,4))
plt.plot(t, r, label="rabbits r(t)")
plt.plot(t, f, label="foxes f(t)")
plt.xlabel("time t")
plt.ylabel("population")
plt.title("Rabbits and foxes vs time (r0=15, f0=22)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: Phase plane (f, r)
# -----------------------------
plt.figure(figsize=(6,5))
plt.plot(f, r)
plt.xlabel("foxes f(t)")
plt.ylabel("rabbits r(t)")
plt.title("Phase plane (f, r) for r0=15, f0=22")
plt.grid(True)
plt.tight_layout()
plt.show()
