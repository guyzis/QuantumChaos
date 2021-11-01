"""
Some examples of executing the src_code and plotting the results
"""
from utils import *
from build_hamiltonian import *
from legacy_utils import *
from static_tools import *
from dynamical_tools import *
from figures_module import *

if __name__ == '__main__':
    jx = 2
    jz = 1
    f = 0.5
    a = 1
    bc = 0

    fig, axs = prepare_standard_figure(nrows=3, ncols=2, tight=True, width=2 * 3.375, aspect_ratio=1.618 * 0.75)
    fig.suptitle(
        r'$J_x={jx},\ J_z={jz},\ \gamma={f},\ \alpha={a},\ B.C={bc}$'.format(jx=jx, jz=jz, f=f, a=a,
                                                                             bc=bc), fontsize=10)

    ns = [10, 12, 14]
    for n in ns:
        basis = blockex(n, n // 2)
        H0 = matt0(n, jx, jz, basis, bc)
        H = matt_add_stark(H0, n, f, a, basis, bc)

        E, V = la.eigh(H.A)

        axs[2, 0].plot(n, rfold(E), 'o')

        var, kurt = offdiag_stats([E, V], n, basis)
        axs[2, 1].plot(n, kurt, 'o')

        # msd should be averaged over multiple runs
        x = msd(H, n, basis, t=25, k=50, dt=0.1)

        axs[0, 0].plot(x[0], x[1] - x[1][0], label=f'$L = {n}$')

        x = diag_elements([E, V], n, basis)
        axs[0, 1].scatter(x[0], x[1], label=f'$L = {n}$', s=1)

        x = offdiag_dist([E, V], n, basis)
        axs[1, 1].step(x[0], x[1], label=f'$L = {n}$')

        x = offdiag([E, V], n, basis)
        axs[1, 0].plot(x[0], x[1], '.', ms=1)

    axs[0, 0].set_ylabel(r'$MSD$')
    axs[0, 0].set_xlabel(r'$time$')

    axs[0, 1].set_ylabel(r'$\hat{O}_{\alpha,\alpha}$')
    axs[0, 1].set_xlabel(r'$E_\alpha$')

    axs[1, 1].set_ylabel(r'$P\left(\hat{O}_{\alpha,\beta}\right)$')
    axs[1, 1].set_xlabel(r'$\hat{O}_{\alpha,\beta}$')
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend(loc='best')

    axs[1, 0].set_ylabel(r'$\Gamma_{\hat{O}}\left(\omega\right)$')
    axs[1, 0].set_xlabel(r'$\omega$')
    axs[1, 0].hlines(np.pi / 2, np.min(x[0]), np.max(x[0]), ls='--', color='k', label=r'$\pi/2$')
    axs[1, 0].legend(loc='best')

    axs[2, 0].set_xlabel(r'$L$ - chain size')
    axs[2, 0].set_xticks(np.arange(ns[0], ns[-1] + 1))
    axs[2, 0].set_ylabel(r'$\langle r \rangle$')
    axs[2, 0].hlines([0.39, 0.536], ns[0], ns[-1], ls='--', color='k')

    axs[2, 1].set_xlabel(r'$L$ - chain size')
    axs[2, 1].set_xticks(np.arange(ns[0], ns[-1] + 1))
    axs[2, 1].set_ylabel(r'$Kurtosis$')
    axs[2, 1].hlines(3, ns[0], ns[-1], ls='--', color='k')

    plt.show()