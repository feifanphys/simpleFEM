import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

def FD(x):
    a = np.exp(-x)
    b = 3*(pi/2)**2 * ((x+2.13) + ((abs(x-2.13))**2.4 + 9.6)**(5/12))**-1.5
    return 1/(a+b)

x = 370*3
print(FD(x))