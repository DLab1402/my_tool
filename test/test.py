<<<<<<< HEAD
import serial
import matplotlib.pyplot as plt
import time

# -------- SERIAL CONFIG --------
PORT = "COM4"
BAUD = 115200
TIMEOUT = 1      # seconds
OUTPUT_FILE = "serial_data.txt"

# -------- OPEN SERIAL --------
ser = serial.Serial(
    port=PORT,
    baudrate=BAUD,
    timeout=TIMEOUT
)

print("Connected to", PORT)

data = []

try:
    with open(OUTPUT_FILE, "w") as f:
        print("Reading serial data... Press Ctrl+C to stop")

        while True:
            line = ser.readline().decode("utf-8").strip()

            if line:  # ignore empty lines
                print(line)
                f.write(line + "\n")

                try:
                    value = float(line)
                    data.append(value)
                except ValueError:
                    pass  # ignore non-numeric lines

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    ser.close()
    print("Serial closed")

# -------- PLOT DATA --------
if data:
    plt.figure()
    plt.plot(data)
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.title("Serial Data Plot")
    plt.grid(True)
    plt.show()
else:
    print("No numeric data to plot")
=======
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
>>>>>>> e8e128823d559ed2d187b1907a57ddd540111b5e
