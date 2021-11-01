The "XXZ" chain can be written in terms of spin matrices using
$$\hat{H}=\sum_{j=1}^{L-1}\frac{J_{xy}}{2}\left(\hat{S}_{j}^{+}\hat{S}_{j+1}^{-} + \hat{S}_{j}^{-}\hat{S}_{j+1}^{+}\right)+J_{z}\hat{S}_{j}^{z}\hat{S}_{j+1}^{z},$$
where $\hat{S}^{\{x,y,z\}}$ are the Pauli matrices times a factor of $\frac{1}{2}$.

Usually, this type of systems can be **classified into two distinct classes**:

1. Many-body localized systems (MBL).
2. Quantum Chaotic systems (thermalizing systems).

Some key differences are:

+------------------------------------+---------------------+-------------------------+
| **Feature**                        | **Quantum Chaotic** | **Many-body Localized** |
+====================================+=====================+=========================+
| Memory of initial conditions       | None                | Some                    |
+------------------------------------+---------------------+-------------------------+
| Eigenvalues                        | Correlated          | Uncorrelated            |
+------------------------------------+---------------------+-------------------------+
| Off-diagonal elements distribution | Normal              | Sharp distribution      |
+------------------------------------+---------------------+-------------------------+
| Transport                          | Super-diffusive     | Sub-diffusive           |
+------------------------------------+---------------------+-------------------------+
| Entanglement entropy spreading     | Power-law           | Logarithmic             |
+------------------------------------+---------------------+-------------------------+

* **Static properties** are usually examined by using the Hamiltonian eigen-basis, where $E_\alpha = \left\langle {\phi_\alpha}\right .\left|{\hat{H}}\right|\left .{\phi_\alpha}\right \rangle$.
* **Dynamical properties** are examined using the time evolution operator $\hat{U}(t)=e^{-i\hat{H}t}$.