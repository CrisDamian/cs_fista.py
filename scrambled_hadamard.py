#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class that implements the Scrambled Hadamard measurement matrix.

@author: cristian
"""

import numpy as np
from numpy.random import default_rng
from fht import fht


class ScrambledHadamard:
    """\
    Implements multiplication with a scrambled Hadamard matrix
    using the fast Walsh-Hadamard transform.
    """

    def __init__(self, n_samples, n_measurements, seed=None):
        "Initializes the matrix with a random permutation of columns"
        rng = default_rng()
        self.n_samples = n_samples
        self.n_measurements = n_measurements
        self.permutation = rng.choice(n_samples, n_samples, replace=False)

    def right_mult(self, x):
        "Multiples a vector to the right: y = Hx"
        x1 = np.empty_like(x)
        x1[self.permutation] = x
        y = fht(x1)
        return y[:self.n_measurements]

    def left_mult(self, y):
        "Multiples a vector to the left: y = xH"
        y1 = np.zeros(self.n_samples, dtype=y.dtype)
        y1[:self.n_measurements] = y
        x1 = fht(y1)
        return x1[self.permutation]



