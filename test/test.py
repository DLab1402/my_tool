arr = [1, 0, 0, 2, 2, 1, 3, 1, 0]

modified_arr = [10 if x in (2,3) else (0 if x not in (0, 10) else x) for x in arr]

print(modified_arr)
