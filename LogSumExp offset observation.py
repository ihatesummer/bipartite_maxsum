import numpy as np

np.set_printoptions(precision=2)

arr = np.random.randn(10)
print(f"array: {arr}")
print(f"max: {np.max(arr)}")
print(f"LogSumExp: {np.log(np.sum(np.exp(arr)))}")
print(f"difference: {np.log(np.sum(np.exp(arr))) - np.max(arr)}")
print(f"log(n): {np.log(np.size(arr))}")
