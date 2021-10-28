import numpy
import scipy
from utils import *
from build_hamiltonian import *
from static_tools import *
from dynamical_tools import *

if __name__ == '__main__':
    n = 10
    ordd = block0(10)
    H = matt_stark(n, 2, 1, 0.5, 3, ordd, 0)
    print(rfold(H.A))
    x = msd99(H, n, ordd, t=20, k=50, dt=0.1)
    plt.errorbar(x[0], x[1])
    plt.show()
