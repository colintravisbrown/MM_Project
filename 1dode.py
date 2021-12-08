import numpy as np
import numba as nb

@nb.jit()
def oregonator(x,z):
    dx = x*(1 - x) - f*z*(x - q)/(q + x)
    dz = x - z
    return dx/e, dz

x0, z0 = (0.5,0.5)

f,q,e = (1.0, 0.015, 0.61)

t0, tf, h = (0,100,0.0001)

t = np.arange(t0,tf,h)

x = np.zeros_like(t)
x[0] = x0
z = np.zeros_like(t)
z[0] = z0
for i in range(len(t)-1):
    dx, dz = oregonator(x[i], z[i])
    x[i+1] = x[i] + h*dx
    z[i+1] = z[i] + h*dz


import matplotlib as mpl
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(t,x)
plt.show()