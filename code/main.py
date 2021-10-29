"""
Some examples of executing the code and plotting the results
"""
from utils import *
from build_hamiltonian import *
from static_tools import *
from dynamical_tools import *
from figures_module import *

if __name__ == '__main__':
    n = 12
    jx = 2
    jz = 1
    f = 0.5
    a = 1
    bc = 0

    ordd = block0(n)
    H = matt_stark(n, jx, jz, f, a, ordd, bc)

    E, V = la.eigh(H.A)
    print('<r> = %1.2f' % rfold(E))

    x = msd(H, n, ordd, t=25, k=50, dt=0.1)

    fig, axs = prepare_standard_figure(nrows=2, ncols=2, tight=True, width=2*3.375)
    fig.suptitle(r'$L={n},\ J_x={jx}, J_z={jz},\ \gamma={f},\ \alpha={a},\ B.C={bc}$'.format(n=n, jx=jx, jz=jz, f=f, a=a, bc=bc), fontsize=10)

    ax = axs[0, 0]
    ax.plot(x[0], x[1], label=f'$L = {n}$')
    ax.set_ylabel(r'$MSD$')
    ax.set_xlabel(r'$time$')
    ax.legend(loc='best')

    x = diag_elements([E, V], n, ordd)
    ax = axs[0, 1]
    ax.scatter(x[0], x[1], label=f'$L = {n}$', s=1)
    ax.set_ylabel(r'$\hat{O}_{\alpha,\alpha}$')
    ax.set_xlabel(r'$E_\alpha$')
    ax.legend(loc='best')

    x = offdiag([E, V], n, ordd)
    ax = axs[1, 0]
    ax.plot(x[0], x[1], '.', label=f'$L = {n}$', ms=1)
    ax.set_ylabel(r'$\Gamma_{\hat{O}}\left(\omega\right)$')
    ax.set_xlabel(r'$\omega$')
    ax.hlines(np.pi / 2, np.min(x[0]), np.max(x[0]), ls='--', color='k')
    ax.legend(loc='best')


    x = offdiag_dist([E, V], n, ordd)
    ax = axs[1, 1]
    ax.step(x[0], x[1], label=f'$L = {n}$')
    ax.set_ylabel(r'$P\left(\hat{O}_{\alpha,\beta}\right)$')
    ax.set_xlabel(r'$\hat{O}_{\alpha,\beta}$')
    ax.legend(loc='best')
    plt.show()
