import matplotlib.pyplot as plt
filename = 'serial_data.txt'
def extract_periods(filename):
    with open(filename, 'r') as f:
        data = f.read()

    # keep only 0 and 1
    data = ''.join(c for c in data if c in '01')
    signal = [int(c) for c in data]

    zero_periods = []
    one_periods = []

    current = data[0]
    count = 1

    for bit in data[1:]:
        if bit == current:
            count += 1
        else:
            if current == '0':
                zero_periods.append(count)
            else:
                one_periods.append(count)

            current = bit
            count = 1

    # last period
    if current == '0':
        zero_periods.append(count)
    else:
        one_periods.append(count)

    return zero_periods, one_periods,signal

zero_periods, one_periods,data = extract_periods(filename)

plt.figure()
plt.step(range(5000), data[:5000], where="post")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.title("Signal – first 20000 samples")
plt.ylim(-0.2, 1.2)
plt.grid(True)
plt.show()

plt.figure()
plt.hist(zero_periods, bins=20)
plt.xlabel("0-period length (samples)")
plt.ylabel("Count")
plt.title("Histogram of 0-periods")
plt.grid(True)

plt.figure()
plt.hist(one_periods, bins=20)
plt.xlabel("1-period length (samples)")
plt.ylabel("Count")
plt.title("Histogram of 1-periods")
plt.grid(True)

plt.show()