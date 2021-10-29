"""
.. include:: statics.md
"""

import numpy as np
import numpy.linalg as la
from numpy.polynomial import polynomial as poly
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
import itertools
import sys

from utils import *
from build_hamiltonian import *


def rfold(H):
    r"""
    A chaos matrix the arises from the eigenvalues spacing statstics

    Defining the level spacing $s_{\alpha}=E_{\alpha+1}-E_{\alpha}$ one can define the 'r-metric' by $$r_{\alpha}=\min\left(\frac{s_{\alpha}}{s_{\alpha-1}},\frac{s_{\alpha-1}}{s_{\alpha}}\right),$$
    where $$\left\langle r\right\rangle \approx\begin{cases}
    0.39 & \text{Poisson dist.}\\
    0.536 & \text{Wigner Dyson dist.}
    \end{cases}$$

    Args:
        H (np.array): matrix or eigenvalues

    Returns:
        (float): $\langle r \rangle$

    """
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


def unfold(H, cut=2000, normed=False):
    """
    Returns the unfolded eigenvalues (used to remove the bais of the denisity of states in the level spacing
    statistics).

    Args:
        H (np.array): a matrix or the eigenvalues of it
        cut (int): how many eigenvales to cut
        normed (bool): normalize the density of staes

    Returns:
        (np.array): unfolded eigenvalues

    """
    if H.ndim == 2:
        E = np.sort(la.eigvalsh(H))
    elif H.ndim == 1:
        E = np.sort(H)

    eps = (E - np.min(E)) / (np.max(E) - np.min(E))
    e_old = (E - np.min(E)) / (np.max(E) - np.min(E))
    y = np.arange(E.shape[0])
    if normed:
        y = y / E.shape[0]
    if cut != 0:
        eps = eps[cut:-cut]
        y = y[cut:-cut]
    c = poly.polyfit(eps, y, 4)
    # plt.step(e_old, np.arange(e_old.shape[0]))
    # plt.plot(eps, poly.polyval(eps, c))
    # plt.plot([e_old[cut], e_old[-cut]], poly.polyval([e_old[cut], e_old[-cut]], c), 'ok')
    # plt.show()
    return poly.polyval(eps, c)


def num_var(H, l_min=1e-1, dots_number=1000, cut=1000, window=1, normed=False):
    """
    *Docs in progress*
    """
    if H.ndim == 2:
        E = la.eigvalsh(H.A)
    elif H.ndim == 1:
        E = np.sort(H)
    E = E[int(E.shape[0] * (1 - window) / 2):int(E.shape[0] * (1 + window) / 2)]
    eps = unfold(E, cut=cut)
    r = np.max(eps) - np.min(eps)
    l = np.logspace(np.log10(l_min * np.mean(np.diff(eps))), np.log10(r / 4), dots_number)
    vars = np.zeros(dots_number)
    ne = np.zeros(dots_number)
    for i in range(0, dots_number):
        hist = np.histogram(eps, bins=int(r / l[i]))[0]
        ne[i] = np.mean(hist)
        vars[i] = np.var(hist)
    if normed:
        return l, vars
    else:
        return ne, vars


def pr_eigen(H0, basis):
    """
    *Docs in progress*
    """
    tz = time.time()
    if isinstance(H0, list):
        e = H0[0]
        v = H0[1]
    else:
        e, v = la.eig(H0.A)
    pr = np.zeros(e.shape[0])
    for i in range(0, e.shape[0]):
        # pr[i] = np.sum(np.abs(v[:, i]) ** 4) ** -1
        pr[i] = np.sum(np.abs(np.dot(basis.T.conjugate(), v[:, i])) ** 4) ** -1
    print('pr_eigen time was: %s' % ptime(tz))
    return e, pr


def diag_elements(H0, n, ord):
    r"""
    ETH diagonal elements test, used to plot $\left\langle {\phi_\alpha}\right .\left|{\hat{O}}\right|\left .{\phi_\alpha}\right \rangle$ as function of the energy $E_\alpha$.

    Args:
        H0 (sparse matrix): the Hamiltonian
        n (int): number of spins
        ord:

    Returns:
        (np.array, np.array): normalized energies (from 0 to 1), corresponding diagonal element

    """
    tz = time.time()
    l = np.flipud(ordtobit(ord, n))
    if isinstance(H0, list):
        e = H0[0]
        v = H0[1]
    else:
        e, v = la.eig(H0.A)
    print("\ndiag time was: %s" % ptime(tz))
    o = matt0sz(int(n / 2) - 1, l)
    d = la.inv(v).dot(o.A).dot(v)
    d = d.diagonal()
    print("diag elements time was: %s\n" % ptime(tz))
    eps = (e - np.min(e)) / (np.max(e) - np.min(e))
    return eps, d


def offdiag(H0, n, ord, dw=0.05, de=0.05, is_unfold=False):
    r"""
    ETH for off-diagonal elements test, used to identify the chaoticity of an Hamiltonian as a function of energy.
    Used to plot how close to gaussian dist are the off-diagonal elements $\hat{O}_{\alpha\beta}={\left.\left\langle
    {\phi_\alpha}\right .\left|{\hat{O}}\right|\left .{\phi_\beta}\right \rangle\right.}_{\alpha\neq\beta}$ as a
    function of the frequency $\omega\equiv E_{\beta} - E_{\alpha}$.

    The gaussianity test is defined as $\Gamma_{\hat{O}}\left(\omega\right)=\frac{\overline{\left|O_{
    \alpha\beta}\right|^{2}}}{\overline{\left|O_{\alpha\beta}\right|}^{2}}$ which is $\pi/2$ for Gaussian distribution.


    Args:
        H0: Hamiltonian or eigenvalues and eigenvectors
        n (int): number of spins
        l:
        dw (float): size of frequency bins
        de (float): partial part of the spectrum that is being examined (from 0 to 1)
        is_unfold (bool): take unfolded eigenvalues

    Returns:
        (np.array, np.array): $\omega$, $\Gamma_{\hat{O}}\left(\omega\right)$

    """
    tz = time.time()
    l = np.flipud(ordtobit(ord, n))
    if isinstance(H0, list):
        e = H0[0]
        v = H0[1]
    else:
        e, v = la.eig(H0.A)
    print("\ndiag time was: %s" % ptime(tz))
    o = matt0sz(int(n / 2) - 1, l)
    eps = np.max(e) - np.min(e)
    emean = np.mean(e)
    s = la.inv(v).dot(o.A).dot(v)
    s = s.flatten()
    e1 = np.tile(e, (l.shape[0], 1))
    ebar = (e1 + e1.T).flatten() / 2
    logic = np.logical_and(ebar < emean + 0.5 * de * eps, ebar > emean - 0.5 * de * eps)
    w_real = np.abs(e1 - e1.T).flatten()
    bin_num = int((np.max(w_real) - np.min(w_real)) / dw)
    if is_unfold:
        e1 = np.tile(unfold(e, cut=0), (l.shape[0], 1))
    w = np.abs(e1 - e1.T).flatten()
    w = w[logic]
    s = s[logic]
    wspace = np.linspace(np.min(w), np.max(w), bin_num)
    sbins = sp.stats.binned_statistic(w, s ** 2, statistic='mean', bins=wspace)[0]
    sbins = sbins / sp.stats.binned_statistic(w, np.abs(s), statistic='mean', bins=wspace)[0] ** 2
    print("offdiag elements time was: %s\n" % ptime(tz))
    return wspace[1:], sbins


def offdiag_dist(H0, n, ord, e_number=200, bin_num=200, normed=False):
    r"""
    ETH for off-diagonal elements test, used to plot the distribution of the off-diagonal elements of some local
    observable. return the histogram of  $\hat{O}_{\alpha\beta}={\left.\left\langle {\phi_\alpha}\right .\left|{\hat{
    O}}\right|\left .{\phi_\beta}\right \rangle\right.}_{\alpha\neq\beta}$ built from ``e_number`` of eigenstates
    near the middle of the spectrum.

    Args:
        H0 (sparse matrix): the Hamiltonian
        n (int): number of spins
        l:
        e_number (int): number of eigenvalues used to calculate the
        bin_num (int): number of bins for the histogram
        normed (bool): normalize by the system size and Hilbert dimension

    Returns:
        (np.array): an array with the histogram x,y data (same number of points at each axis)

    """
    tz = time.time()
    if isinstance(H0, list):
        e = H0[0]
        v = H0[1]
    else:
        e, v = la.eig(H0.A)
    print("\ndiag time was: %s" % ptime(tz))
    l = np.flipud(ordtobit(ord, n))
    dim = e.shape[0]
    o = matt0sz(int(n / 2) - 1, l)
    s = la.inv(v).dot(o.A).dot(v)
    logic = np.arange(int((e.shape[0] - e_number) / 2), int((e.shape[0] + e_number) / 2))
    e1 = np.tile(e, (l.shape[0], 1))
    ebar = (e1 + e1.T) / 2
    w = (e1 - e1.T)
    s = s[np.ix_(logic, logic)]
    s = s[np.triu_indices(s.shape[0], k=1)] * dim ** (0.5 * int(normed))
    print('mean(ebar) = ', np.mean(ebar[logic]))
    print('mean(w) = ', np.mean(w[logic]))
    print(s.shape)
    hist = np.histogram(s.flatten(), bins=bin_num, density=True)
    print("offdiag_dist elements time was: %s\n" % ptime(tz))
    return np.array([hist[1][1:], hist[0]])
