"""
Dynamics

.. include:: dynamics.md
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
# import h5py
# import sympy as sym
import itertools
# from sympy.core import sympify
import sys

from utils import *
from build_hamiltonian import *


def matt3sz(n, i, ord):
    if i < 0:
        return 1 / 0
    tz = time.time()
    # H = np.zeros([ord.shape[0], ord.shape[0]])
    # H = spr.dok_matrix((ord.shape[0], ord.shape[0]))
    # H = spr.dia_matrix((ord.shape[0], ord.shape[0]))
    h = np.zeros(ord.shape[0])
    l = np.flipud(ordtobit(ord, n))
    h = h + (l[:, i] - 0.5)
    # print("L = %i matt2sz time was %1.3f" %(n, time.time() - tz))
    return spr.dia_matrix((h, 0), shape=(ord.shape[0], ord.shape[0]))


def evolkryl3(H, v, dt, T):
    for i in range(0, T):
        v = las.expm_multiply(-1j * H * dt, v)
    return v


def msd99(H, n, ord, t, k, dt, seed=False, neel=False):
    tz1 = time.time()
    tt = np.linspace(0, t, k)
    rng = range(int(- n / 2) + 1, int(n / 2) + 1)
    grprofile = np.zeros([k, len(rng)])
    i = int(n / 2) - 1  # warning!!!
    op = matt3sz(n, i, ord)
    if seed:
        np.random.seed(1)
    v = np.random.normal(0, 1, ord.shape[0]) + 1j * np.random.normal(0, 1, ord.shape[0])
    if neel:
        v = np.zeros(ord.shape[0])
        nl = np.mod(np.arange(n) + 1, 2)
        v[int(np.argwhere(bittoint(nl) == np.flip(block0(n))))] = 1
    psi = v / la.norm(v)
    psi2 = op.dot(psi)
    mss = np.zeros(k)
    for l in range(0, k):
        ms = 0
        for j in range(0, len(rng)):
            r = rng[j]
            op2 = matt3sz(n, i + r, ord)
            grt = np.real(np.dot(np.conj(psi), op2.dot(psi2.T)))
            ms = ms + r ** 2 * grt
            grprofile[l, j] = grt
        if l < k - 1:
            T = np.floor((tt[l + 1] - tt[l]) / dt).astype(int)  # number of steps in the section
            dt2 = (tt[l + 1] - tt[l]) / T  # fine tuning dt so it will actually fit the number of steps
            psi = evolkryl3(H, psi, dt2, T)
            psi2 = evolkryl3(H, psi2, dt2, T)
        mss[l] = ms
    print("L = %i, k = %i ,one haar measure time: %s" % (n, k, ptime(tz1)))
    return (tt, mss, grprofile)