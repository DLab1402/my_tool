import numpy as np

arr = np.array([[[1, 5, 3], [7, 2, 8]]])
print(arr)

# Find the index of the max value along axis 1 (row-wise)
max_indices_row = np.argmax(arr, axis=-2)
print("Max indices along rows:", max_indices_row)
