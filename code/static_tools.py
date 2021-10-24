"""
.. include:: statics.md
"""

import numpy as np
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp
from scipy import sparse as spr
from scipy.sparse import linalg as las
import scipy.interpolate as intr
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.linalg import expm, sinm, cosm
from scipy import optimize
from scipy import stats
import os
# from sympy.utilities.iterables import multiset_permutations
import h5py
# import sympy as sym
import itertools
# from sympy.core import sympify
import sys

from utils import *



def rfold(H):
    t = time.time()
    if H.ndim == 2:
        E = np.sort(la.eigh(H)[0])
    elif H.ndim == 1:
        E = H
    # print("Diagonalization time was: ", time.time() - t)
    S = np.diff(E)
    r = 0
    for i in range(1, S.shape[0]):
        r = r + min(S[i], S[i - 1]) / max(S[i], S[i - 1])
    r = r / (S.shape[0] - 1)
    return r