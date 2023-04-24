import matplotlib.pyplot as plt
import numpy as np
from math import exp, log

def f(b):
    tau = -log(0.5)/0.3
    return 1 - np.e**(-tau*b)

x = np.linspace(0, 1, 1000)
y = f(x)
plt.figure()
# plt.grid()
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.plot(x,  y)
plt.ylabel('QoE Score Weight')
plt.xlabel('Buffer Occupancy')
plt.show()
