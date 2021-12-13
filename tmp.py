import numpy as np
import tensorflow as tf

a = np.array([1,2,3])
b = np.append(a, a)
c = np.reshape(b, (2, 3))
print(c)