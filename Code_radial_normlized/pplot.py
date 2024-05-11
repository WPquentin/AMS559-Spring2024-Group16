from __future__ import division
import matplotlib as mpl, matplotlib.pyplot as plt
import time
import pickle
import numpy as np
import diffusion_scheme as dif, plotting as pl

def phi_pos_eq(w):
    a = [4.558, 1.57, 0.058, 0.621]
    b = [0.154, 0.861, 0.888, 0.941]
    c = [0.749, 0.328, 0.0250, 0.148]
    Up = 0
    for i in range(4):
        Up += a[i] * np.exp(-((w-b[i])/c[i])**2)
    return Up

w = np.linspace(0, 1, 100)
y = np.zeros_like(w)
for i in range(100):
    y[i] = phi_pos_eq(w[i])

plt.plot(w,y)
plt.show()
