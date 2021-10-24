\newcommand{\ket}[1]{\left|{#1}\right\rangle}
\newcommand{\bra}[1]{\left\langle{#1}\right|}
\newcommand{\braket}[2]{\left.\left\langle {#1}\right|{#2}\right\rangle}
\newcommand{\obracket}[3]{\left\langle {#1}\right|{#2}\left|{#3}\right\rangle}
\newcommand{\proj}[2]{\left.\left.\left|{#1}\right\rangle \right\langle{#2}\right|}

The "XXZ" chain can be written in terms of spin matrices using
$$\hat{H}=\sum_{j=1}^{L-1}\frac{J_{xy}}{2}\left(\hat{S}_{j}^{x}\hat{S}_{j+1}^{x} + \hat{S}_{j}^{y}\hat{S}_{j+1}^{y}\right)+J_{z}\hat{S}_{j}^{z}\hat{S}_{j+1}^{z},$$
where $\hat{S}^{\{x,y,z\}}$ are the Pauli matrices times a factor of $\frac{1}{2}$.

Usually, this type of systems can be classified into two classes:

1. Many-body localized systems (MBL).
2. Quantum Chaotic systems ( thermalizing systems).

Some key differences are:

Feature | Quantum Chaotic   | Many-body localized |
| :----------- | ----------- | ----------- |
Memory of initial conditions | None   | Some |
Eigenvalues | Correlated  | Uncorrelated  |
Off-diagonal elements distribution | Normal  |  Sharp distribution |
Transport | Super-diffusive  | Sub-diffusive  |
Entanglement entropy spreading | Power-law  | Logarithmic  |

* **Static properties** are usually examined by using the Hamiltonian eigen-basis, where $E_\alpha = \obracket{\phi_\alpha}{\hat{H}}{\phi_\alpha}$.
* **Dynamical properties** are usually examined by using the time evolution operator $\hat{U}(t)=e^{-i\hat{H}t}$.