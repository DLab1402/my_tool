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
