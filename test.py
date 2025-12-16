import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.001
T = 8
time = np.arange(0, T, dt)
N = len(time)

# Second-order plant: x'' + 2ζx' + x = u
# Convert to state space: x1'=x2, x2' = -2ζ x2 - x1 + u
def simulate_pid(Kp=0, Ki=0, Kd=0, zeta=0.3):
    x1 = 0.0  # output
    x2 = 0.0  # velocity
    integral = 0.0
    prev_e = 0.0
    y_hist = np.zeros(N)
    r = 1.0

    for i, t in enumerate(time):
        e = r - x1
        integral += e * dt
        derivative = (e - prev_e) / dt
        prev_e = e

        u = Kp * e + Ki * integral + Kd * derivative

        # Update plant
        x1_new = x1 + dt * x2
        x2_new = x2 + dt * (-2*zeta*x2 - x1 + u)

        x1, x2 = x1_new, x2_new
        y_hist[i] = x1

    return y_hist

# Cases:
y_good = simulate_pid(Kp=2, Ki=1, Kd=0.2, zeta=0.7)      # Good stable response
y_overshoot = simulate_pid(Kp=6, Ki=2, Kd=0, zeta=0.3)   # Overshoot
y_oscillate = simulate_pid(Kp=10, Ki=0, Kd=0, zeta=0.1)  # Strong oscillation (underdamped)

plt.figure(figsize=(8,4))
plt.plot(time, y_good, label="Đáp ứng tốt (PID)")
plt.plot(time, y_overshoot, label="Quá điều chỉnh (overshoot)")
# plt.plot(time, y_oscillate, label="Dao động mạnh (lọt võ/vượt biên)")

plt.axhline(1.0, linestyle='--', linewidth=0.8)
plt.xlabel("Thời gian (s)")
plt.ylabel("Đầu ra y(t)")
plt.title("Các dạng đáp ứng: ổn định – quá điều chỉnh – dao động mạnh")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
