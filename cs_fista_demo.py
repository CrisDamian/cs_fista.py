# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:56:39 2018

@author: DamianCristian

cs_fista_demo.py

Usage demonstration of cs_fista.py

"""

import numpy as np
from skimage import data
from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt
plt.rcParams['image.cmap']='gray'
from time import process_time as clock

from scrambled_hadamard import ScrambledHadamard
from cs_fista import Fista

def total_variation(X):
    """
    Computes the Total Variation of an array.
    """
    h=[-1,1]
    g = np.zeros_like(X)
    for i in range(0,X.ndim):
        g += convolve1d(X,h,axis=i)**2
    return np.sum(g**.5)

n = 128**2
m = int(0.2 * n);

print("Number of samples =",n)
print("Number of measurements =",m)

# Original Image
ishape = (int(n**.5),)*2
I = data.binary_blobs(ishape[0])
I = np.asarray(I, dtype=float)
print('Image TV=', total_variation(I))

# Measurement matrix
H = ScrambledHadamard(n, m)

# Defining measurement operator and adjoint
p = lambda x: H.right_mult(x.flat[:])
pt = lambda x: np.reshape(H.left_mult(x), ishape)

# Measurements
D = p(I)

# Least Squares estimate
t1 = clock()
e0 = pt(D)
E0 = np.reshape(e0, ishape)
runtime = clock()-t1

print('Naive Runtime=', runtime)
print('Naive PSNR=', 10*np.log10(1/np.var(E0-I)))

# Total Variation CS estimate
solver = Fista(p, 'tv', pt, 1e-2)
t1 = clock()
E = solver.reconstruct(D)
runtime = clock()-t1

print('TV Runtime=', runtime)
print('TV PSNR=', 10*np.log10(1/np.var(E-I)))  

f, s = plt.subplots(1,3)
s[0].imshow(E0)
s[0].set_title("LS reconstrucion")
s[1].imshow(E)
s[1].set_title("TV reconstrucion")
s[2].imshow(I)
s[2].set_title("Original Image")
plt.show()
