import numpy
import scipy
from utils import *
from build_hamiltonian import *
from static_tools import *
from dynamical_tools import *
from figures_module import *

if __name__ == '__main__':
    n = 10
    ordd = block0(10)
    H = matt_stark(n, 2, 1, 0.5, 3, ordd, 0)
    print(rfold(H.A))
    x = msd(H, n, ordd, t=20, k=50, dt=0.1)

    fig, ax = prepare_standard_figure(tight=True)
    ax.plot(x[0], x[1], label=f'$L = {n}$')
    ax.set_ylabel(r'$MSD$')
    ax.set_xlabel(r'$time$')
    ax.legend(loc='best')
    plt.show()
