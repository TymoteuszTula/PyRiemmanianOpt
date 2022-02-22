# riemann_grad_unit_opt_renyi.py

r"""This code contains the Riemmanian SD algorithm, designed to optimise a cost function J(U)
defined on the unitary matrices space. Cost function is Renyi entropy.

References
----------
    [1] Traian E. Abrudan, Jan Eriksson, and Visa Koivunen. 2008. Steepest descent algorithms for 
    optimization under unitary matrix constraint. IEEE Transactions on Signal Processing, volume 56,
    number 3, pages 1134-1147. © 2008
    [2] Traian Abrudan, Jan Eriksson, and Visa Koivunen. 2008. Conjugate gradient algorithm for 
    optimization under unitary matrix constraint. Helsinki University of Technology, Department of 
    Signal Processing and Acoustics, Report 4. ISBN 978-951-22-9483-1. ISSN 1797-4267. 
    Submitted for publication. © 2008 
    [3] Traian Abrudan, Jan Eriksson, and Visa Koivunen. 2007. Efficient line search methods for 
    Riemannian optimization under unitary matrix constraint. In: Conference Record of the 
    Forty-First Asilomar Conference on Signals, Systems and Computers (ACSSC 2007). Pacific Grove, 
    CA, USA. 4-7 November 2007, pages 671-675. © 2007

"""

import numpy as np
from scipy.linalg import inv 
from math import pi
from tenpy.linalg.np_conserved import expm, tensordot, eigvals, Array, trace, norm, svd

eps = 1.e-12

class RiemannGradUnitOptRenyi:
    r"""A class containing functions for calculating renyi entropies, euclidian derivatives of renyi
    entropies and various line search methods. After initialisation, self.run_opt can be run to 
    perform optimization.
    
    Parameters
    ----------
    W0 : :class: `~tenpy.linalg.np_conserved.Array`, (n, n, n, n)
        Initial point for optimization algorithm. Usually it can be a Idenity matrix, but could be 
        set to other if there is a better initial guess. Has to be unitary.
        (NOTE Currently it is assumed that W0 should be a 4-tensor with legs 
        ['q0', 'q0*', 'q1', 'q1'*], which after joining '(q0.q1)', '(q0*.q1*)' transforms into 
        unitary matrix, i. e. ::
            
            |      q0* q1* 
            |       ^   ^
            |       |   |             |   |       |   |
            |       |---|             |---|       |   |
            |       | W |             | W |       |   |
            |       |---|             |---|       |   |
            |       |   |             |   |   =   |   |
            |       ^   ^             |   |       |   |
            |      q0  q1             |---|       |   |
            |                         | W*|       |   |
            |                         |---|       |   |
            |                         |   |       |   |

        )
    grad_method : {'sdsa', 'cgpr', 'cgfr'}
        Specifies different gradients used by the algorithm. 'sdsa' is standard steepest descent
        (following geodesic), and 'cgpr' and 'cgfr' specify conjugate gradient algorithms [reference].
    goed_search_method : {'a', 'p', 'f'}
        Specifies different goedesic step size calculations [reference]. 'a' is an armijo search,
        'p' is a polynomial method and 'f' is Discrete Fourier Series (DFT) method.
    opt : {'min', 'max'}
        Switch changing whether cost function should be minimised or maximised.
    K_iter: `int`
        Number of steps for optimization
    q : `int`
        Coefficient affecting window of search for step size given by 
        $T_\mu = \frac{2\pi}{q|\om_{max}|}$. [reference]
        (NOTE I have not yet derived what it should be for Renyi entropy, I'm recomending setting
        it to 1, and maybe change it to see if optimization is faster).
    alpha : `float` or `int`
        Alpha coefficient in Renyi entropy.

    Attributes
    ----------
    W0 : :class: `~tenpy.linalg.np_conserved.Array`, (n, n, n, n)
        Initial point for optimization algorithm. Usually it can be a Idenity matrix, but could be 
        set to other if there is a better initial guess. Has to be unitary.
        (NOTE Currently it is assumed that W0 should be a 4-tensor with legs 
        ['q0', 'q0*', 'q1', 'q1'*], which after joining '(q0.q1)', '(q0*.q1*)' transforms into 
        unitary matrix, i. e. ::
            
            |      q0  q1 
            |       ^   ^
            |       |   |             |   |       |   |
            |       |---|             |---|       |   |
            |       | W |             | W |       |   |
            |       |---|             |---|       |   |
            |       |   |             |   |   =   |   |
            |       ^   ^             |   |       |   |
            |      q0* q1*            |---|       |   |
            |                         | W*|       |   |
            |                         |---|       |   |
            |                         |   |       |   |

        )
    grad_method : {'sdsa', 'cgpr', 'cgfr'}
        Specifies different gradients used by the algorithm. 'sdsa' is standard steepest descent
        (following geodesic), and 'cgpr' and 'cgfr' specify conjugate gradient algorithms [reference].
    goed_search_method : {'a', 'p', 'f'}
        Specifies different goedesic step size calculations [reference]. 'a' is an armijo search,
        'p' is a polynomial method and 'f' is Discrete Fourier Series (DFT) method.
    opt : {'min', 'max'}
        Switch changing whether cost function should be minimised or maximised.
    K_iter: `int`
        Number of steps for optimization
    n : `int`
        Dimensionality of legs 'q0', 'q0*', 'q1', 'q1*.'
    q : `int`
        Coefficient affecting window of search for step size given by 
        $T_\mu = \frac{2\pi}{q|\om_{max}|}$. [reference]
        (NOTE I have not yet derived what it should be for Renyi entropy, I'm recomending setting
        it to 1, and maybe change it to see if optimization is faster).
    alpha : `float` or `int`
        Alpha coefficient in Renyi entropy.
    theta : :class: `~tenpy.linalg.np_conserved.Array`, (NL, NR, n, n, n, n)
        Placeholder for a given two-site tensor, for which the disentangler that optimises Renyi 
        entropy should be applied.
        (NOTE Currently it is assumed that theta is 6-tensor with legs 
        ['vL', 'vR', 'p0', 'p1', 'q0', 'q1'], so two virtual, two physical and two auxillary legs,
        with disentangler acting just on auxillary part).
    sign_mu: `int`
        sign of step (depends on choosing minimization or maximization).

    """

    def __init__(self, W0, grad_method, geod_search_method, opt, K_iter, q, alpha):
        self.W0 = W0
        self.grad_method = grad_method
        self.geod_search_method = geod_search_method
        self.opt = opt
        self.K_iter = K_iter
        self.test_sanity()
        self.n = self.W0.shape[0]
        self.q = q
        self.alpha = alpha
        self.theta = None
        if opt == 'min':
            self.sign_mu = -1
        elif opt == 'max':
            self.sign_mu = 1
        else:
            raise ValueError("Valid strings for parameter opt: `min` or `max`.")

    def test_sanity(self):
        # TODO write proper tests
        pass

    def renyi_ent(self, Wk):
        r"""Calculate Renyi entropy for combined tensors Wk and self.theta. Cut is taken between
        site 0 and 1.

        Parameters
        ----------
        Wk : :class: `~tenpy.linalg.np_conserved.Array`, (self.n ** 2, self.n ** 2)
            Unitary matrix for which the Renyi entropy is calculated.
            (NOTE Currently, expected legs are (q0.q1), (q0*.q1*))

        Returns
        -------
        float
            Renyi entropy of Wk and self.theta.

        """

        Wk_split = Wk.split_legs()
        # Combine matrix Wk with theta
        mat = tensordot(Wk_split, self.theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        # Join legs from the same sites
        mat = mat.combine_legs([['vL', 'q0', 'p0'], ['vR', 'q1', 'p1']])
        # Perform svd to get singular values
        _, s, _ = svd(mat)
        return np.real(1 / (1 - self.alpha) * np.log(np.sum(s ** self.alpha)))

    def renyi_ent_euc_grad(self, Wk):
        r"""Calculate Euclidian derivatives of Renyi entropy with respect to W at Wk.

        Parameters
        ----------
        Wk : :class: `~tenpy.linalg.np_conserved.Array`, (self.n ** 2, self.n ** 2)
            Unitary matrix at which the Euclidian derivatives of Renyi entropy are calculated.
            (NOTE Currently, expected legs are (q0.q1), (q0*.q1*))

        Returns
        -------
        :class: `~tenpy.linalg.np_conserved.Array`, (self.n ** 2, self.n ** 2)
            Euclidian gradients of Renyi entropies. 
    
        """

        # Find matrices u, v and singular values using svd.
        Wk_split = Wk.split_legs()
        mat = tensordot(Wk_split, self.theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        mat = mat.combine_legs([['vL', 'q0', 'p0'], ['vR', 'q1', 'p1']])
        u, s, v = svd(mat, inner_labels=['(vL*.q0*.p0*)', '(vR*.q1*.p1*)'])

        u_split = u.split_legs()
        v_split = v.split_legs()
        u_split.ireplace_label('q0', 'q0*')
        v_split.ireplace_label('q1', 'q1*')

        deriv = tensordot(u_split, self.theta, axes = [['vL', 'p0'], ['vL', 'p0']])
        deriv = tensordot(v_split, deriv, axes = [['vR', 'p1'], ['vR', 'p1']])

        s_alpha = s ** (self.alpha - 1)
        deriv.iscale_axis(s_alpha, axis='(vR*.q1*.p1*)')

        U_deriv = trace(deriv, leg1='(vR*.q1*.p1*)', leg2='(vL*.q0*.p0*)')

        U_deriv.ireplace_labels(['q0', 'q1', 'q0*', 'q1*'], ['q0*', 'q1*', 'q0', 'q1'])

        traceS = np.sum(s ** self.alpha)

        U_deriv = U_deriv.combine_legs([['q0', 'q1'], ['q0*', 'q1*']])

        return 1 / (1 - self.alpha) * self.alpha / traceS * U_deriv / 2

    def run_opt(self, theta, inplace = False):
        r"""Run optimization for a given theta tensor.

        Parameters
        ----------
        theta : :class: `~tenpy.linalg.np_conserved.Array`, (NL, NR, n, n, n, n)
            Two-site tensor, for which the disentangler that optimises Renyi entropy should be applied.
            (NOTE Currently it is assumed that theta is 6-tensor with legs 
            ['vL', 'vR', 'p0', 'p1', 'q0', 'q1'], so two virtual, two physical and two auxillary legs,
            with disentangler acting just on auxillary part).
        inplace : `bool`
            True if after optimization theta should be changed to theta_new, False if not.

        Returns
        -------
        Wk : :class: `~tenpy.linalg.np_conserved.Array`, (n, n, n, n)
            Final disentangler.
        """

        self.theta = theta

        # TODO write test_sanity_theta

        Wk = self.W0.copy()
        Wk = Wk.combine_legs([['q0', 'q1'], ['q0*', 'q1*']])
        
        for k in range(self.K_iter):
            # Calculate Euclidian derivatives
            if k % self.n**2 == 0:
                GEk = self.renyi_ent_euc_grad(Wk)
                GEkWk = tensordot(GEk, Wk.transpose().complex_conj(), axes=[['(q0*.q1*)'], ['(q0.q1)']])
                Gk = GEkWk - GEkWk.transpose().complex_conj()
                Hk = Gk
            else:
                Gk = Gkplus1
                Hk = Hkplus1 
            # Choose geodesic (line) search method
            N_poly = 5
            NT = 15
            par2 = {'a': Gk, 'p': Hk, 'f': Hk}
            par3 = {'a': Hk, 'p': N_poly, 'f': NT}
            geod_method = self._av_geod_search_methods[self.geod_search_method].__get__(self, type(self))
            mu = geod_method(Wk, par2[self.geod_search_method], par3[self.geod_search_method], self.opt)
            R1 = expm(self.sign_mu * mu * self.skew(Hk))
            # Update
            Wkplus1 = tensordot(R1, Wk, axes=[['(q0*.q1*)'], ['(q0.q1)']])
            GEkplus1 = self.renyi_ent_euc_grad(Wkplus1)
            GEk1Wk1 = tensordot(GEkplus1, Wkplus1.conj(), axes=[['(q0*.q1*)'], ['(q0.q1)']])
            Gkplus1 = GEk1Wk1 - GEk1Wk1.transpose().complex_conj()
            if self.grad_method == 'sdsa':
                gamma_k = 0
            elif self.grad_method == 'cgpr':
                gamma_k = self.innerprod(Gkplus1-Gk, Gkplus1)/ self.innerprod(Gk, Gk)
            elif self.grad_method == 'cgfr':
                gamma_k = self.innerprod(Gkplus1, Gkplus1)/ self.innerprod(Gk, Gk)
            else:
                raise ValueError('grad_method must be one of the following: `sdsa`, `cgpr`, `cgfr`')
            Hkplus1 = Gkplus1 + gamma_k * Hk
            # test if the search direction is wrong
            if self.innerprod(Gkplus1, Hkplus1) < 0:
                Hkplus1 = Gkplus1
            Wk = Wkplus1
        Wk = Wk.split_legs()
        if inplace:
            theta = tensordot(theta, Wk, axes=[['q0', 'q1'], ['q0*', 'q1*']])
        return Wk

    def geod_search_armijo(self, Wk, Gk, Hk, opt):
        if opt not in ['min', 'max']:
            raise ValueError('`opt` must be either `max` or `min`')
        mu = 1
        if opt == 'min':
            R1 = expm(-mu * self.skew(Hk))
            R2 = tensordot(R1, R1, axes=[['(q0*.q1*)'], ['(q0.q1)']])
            J = self.renyi_ent(Wk)
            innGkHk = self.innerprod(Gk, Hk)
            J2 = self.renyi_ent(tensordot(R2, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
            J1 = self.renyi_ent(tensordot(R1, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
            if (J - J2 >= mu * innGkHk) and (J - J1 < (1/2) * mu * innGkHk):
                pass
            else:
                while J - J2 >= mu * innGkHk:
                    mu = 2 * mu
                    R1 = R2
                    R2 = tensordot(R1, R1, axes=[['(q0*.q1*)'], ['(q0.q1)']])
                    J2 = self.renyi_ent(tensordot(R2, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
                J1 = self.renyi_ent(tensordot(R1, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
                while J - J1 < (1/2) * mu * innGkHk:
                    mu = 1/2 * mu
                    R1 = expm(-mu * self.skew(Hk))
                    J1 = self.renyi_ent(tensordot(R1, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
        elif opt == 'max':
            R1 = expm(mu * self.skew(Hk))
            R2 = tensordot(R1, R1, axes=[['(q0*.q1*)'], ['(q0.q1)']])
            J = self.renyi_ent(Wk)
            innGkHk = self.innerprod(Gk, Hk)
            J2 = self.renyi_ent(tensordot(R2, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
            J1 = self.renyi_ent(tensordot(R1, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
            if (J - J2 <= -mu * innGkHk) and (J - J1 > (1/2) * mu * innGkHk):
                pass
            else:
                while J - J2 <= -mu * innGkHk:
                    mu = 2 * mu
                    R1 = R2
                    R2 = tensordot(R1, R1, axes=[['(q0*.q1*)'], ['(q0.q1)']])
                    J2 = self.renyi_ent(tensordot(R2, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
                J1 = self.renyi_ent(tensordot(R1, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
                while J - J1 > -(1/2) * mu * innGkHk:
                    mu = 1/2 * mu
                    R1 = expm(mu * self.skew(Hk))
                    J1 = self.renyi_ent(tensordot(R1, Wk, axes=[['(q0*.q1*)'],['(q0.q1)']]))
        return mu

    def geod_search_poly(self, Wk, Hk, N_poly, opt):
        if N_poly not in [3, 4, 5]:
            raise ValueError('`N_poly` must be either 3, 4 or 5.')
        sign_mu = 1 * (opt == 'max') - 1 * (opt == 'min')
        omega_max = np.max(np.abs(eigvals(Hk)))
        T_mu = 2 * pi / (self.q * omega_max)

        mu_step_poly = T_mu / N_poly
        R_poly = expm(sign_mu * mu_step_poly * Hk)
        d1_poly = np.zeros(N_poly+1)
        R_mu_poly_ndarray = np.eye(Wk.shape[0])
        R_mu_poly_ndarray = R_mu_poly_ndarray.reshape([self.n, self.n, self.n, self.n]) 
        R_mu_poly = Array.from_ndarray_trivial(R_mu_poly_ndarray, labels=['q0', 'q1', 'q0*', 'q1*'])
        R_mu_poly = R_mu_poly.combine_legs([['q0', 'q1'], ['q0*', 'q1*']])
        for n_poly in range(N_poly+1):
            Dk = self.renyi_ent_euc_grad(tensordot(R_mu_poly, Wk, axes=[['(q0*.q1*)'], ['(q0.q1)']]))
            M_trace = tensordot(Dk, Wk.conj(), axes=[['(q0*.q1*)'], ['(q0.q1)']])
            M_trace = tensordot(M_trace, R_mu_poly.conj(), axes=[['(q0*.q1*)'], ['(q0.q1)']])
            M_trace = tensordot(M_trace, Hk.conj(), axes=[['(q0*.q1*)'], ['(q0.q1)']])
            d1_poly[n_poly] = -2 * sign_mu * np.real(trace(M_trace))
            R_mu_poly = tensordot(R_mu_poly, R_poly, axes=[['(q0*.q1*)'], ['(q0.q1)']])

        C = np.array([[] for _ in range(N_poly)])
        mu_poly = np.linspace(0, T_mu, N_poly+1)
        for n_poly in range(1, N_poly + 1):
            C = np.c_[C, mu_poly[1:]**n_poly]
        
        d_new = d1_poly[1:] - d1_poly[0]
        d_new[-1] = 0
        a = np.r_[d1_poly[0], inv(C) @ d_new]
        all_roots_d1 = np.roots(a)
        index_real_roots_d1 = np.where(np.abs(np.imag(all_roots_d1)) <= eps)
        real_roots_d1 = all_roots_d1[index_real_roots_d1]
        index_positive_roots_d1 = np.where(real_roots_d1 > 0)
        return np.min(real_roots_d1[index_positive_roots_d1])

    def geod_search_dft(self, Wk, Hk, NT, opt):
        if NT <= 0:
            raise ValueError('Bad options for parameter NT, must be strictly positive natural number.')
        sign_mu = 1 * (opt == 'max') - 1 * (opt == 'min')
        omega_max = np.max(np.abs(eigvals(Hk)))
        T_mu = 2 * pi / (self.q * omega_max)
        K = 3

        T_dft = NT * T_mu
        N_dft = int(2 * np.floor(K * NT / 2) + 1)
        R_dft = expm(sign_mu * (T_dft/N_dft) * Hk)
        R_mu_dft_ndarray = np.eye(Wk.shape[0])
        R_mu_dft_ndarray = R_mu_dft_ndarray.reshape([self.n, self.n, self.n, self.n])
        R_mu_dft = Array.from_ndarray_trivial(R_mu_dft_ndarray, labels=['q0', 'q1', 'q0*', 'q1*'])
        R_mu_dft = R_mu_dft.combine_legs([['q0', 'q1'], ['q0*', 'q1*']])
        J_dft = np.zeros(N_dft)
        d1_dft = np.zeros(N_dft)
        for n_dft in range(N_dft):
            Dk = self.renyi_ent_euc_grad(tensordot(R_mu_dft, Wk, axes=[['(q0*.q1*)'], ['(q0.q1)']]))
            M_trace = tensordot(Dk, Wk.conj().transpose(), axes=[['(q0*.q1*)'], ['(q0.q1)']])
            M_trace = tensordot(M_trace, R_mu_dft.conj().transpose(), axes=[['(q0*.q1*)'], ['(q0.q1)']])
            M_trace = tensordot(M_trace, Hk.conj().transpose(), axes=[['(q0*.q1*)'], ['(q0.q1)']])
            d1_dft[n_dft] = -2 * sign_mu * np.real(trace(M_trace))
            J_dft[n_dft] = self.renyi_ent(tensordot(R_mu_dft, Wk, axes=[['(q0*.q1*)'], ['(q0.q1)']]))
            R_mu_dft = tensordot(R_mu_dft, R_dft, axes=[['(q0*.q1*)'], ['(q0.q1)']])

        h_window = np.hanning(len(d1_dft)+2)[1:-1]
        d1_dft = d1_dft * h_window
        coefs_dft = np.fft.fftshift(np.fft.fft(d1_dft)) / N_dft
        poly_roots = np.roots(coefs_dft)
        cicle_eps = 10.e-2
        poly_roots = poly_roots[np.abs(np.abs(poly_roots) - 1) < cicle_eps]
        log_poly_roots = np.log(poly_roots)
        imag_log_poly_roots = -np.imag(log_poly_roots)

        mu_roots = (imag_log_poly_roots / (2 * pi / T_dft)) % T_dft
        sorted_mu = np.sort(mu_roots)
        mu_opt_dft = sorted_mu[range(0, len(sorted_mu), 2)]
        if mu_opt_dft.size == 0 or abs(d1_dft[0]) <= 1.e-12:
            mu_opt_dft = 0
        else:
            mu_dft = np.linspace(0, NT * T_mu, N_dft+1)
            mu_dft = mu_dft[:-1]
            if opt == 'min':
                mu_dft_min = mu_dft[np.argmin(J_dft)]
                mu_opt_dft = mu_opt_dft[np.argmin(np.abs(mu_dft_min - mu_opt_dft))]
            else:
                mu_dft_max = mu_dft[np.argmax(J_dft)]
                mu_opt_dft = mu_opt_dft[np.argmin(np.abs(mu_dft_max-mu_opt_dft))]
        return mu_opt_dft

    _av_geod_search_methods = {'a': geod_search_armijo, 'p': geod_search_poly, 'f': geod_search_dft}

    def unit_crit_eval(self, Wk):
        eye = np.eye(self.n ** 2)
        eye = Array.from_ndarray_trivial(eye, labels=['(q0.q1)', '(q0*.q1*)'])
        return norm(tensordot(Wk.conj(), Wk, axes=[['(q0*.q1*)'], ['(q0.q1)']]) - eye)**2

    def skew(self, Hk):
        return 0.5 * (Hk - Hk.transpose().complex_conj())

    def innerprod(self, G1, G2):
        return 0.5 * np.real(trace(tensordot(G1.conj(), G2, axes=[['(q0*.q1*)'], ['(q0.q1)']])))