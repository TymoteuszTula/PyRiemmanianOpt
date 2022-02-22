# riemann_grad_unit_opt.py

r"""This code contains the Riemmanian SD algorithm, designed to optimise a cost
function J(U) defined on the unitary matrices space."""

import numpy as np
from scipy.linalg import expm, eigvals, inv
from math import pi
norm = np.linalg.norm

eps = 1.e-12

class RiemannGradUnitOpt:

    def geod_search_armijo(self, Wk, Gk, Hk, opt):
        if opt not in ['min', 'max']:
            raise ValueError('`opt` must be either `max` or `min`')
        mu = 1 * (opt == 'max') - 1 * (opt == 'min')
        mu_c = {'min': -1, 'max': 1}
        R1 = expm(mu * self.skew(Hk))
        R2 = R1 @ R1
        J = self.cf_eval(Wk)
        innGkHk = self.innerprod(Gk, Hk)
        if (J - self.cf_eval(R2 @ Wk) >= mu * mu_c[opt] * innGkHk) and (J - self.cf_eval(R1 @ Wk) < (1/2)*mu*mu_c['min']*innGkHk):
            pass
        else:
            while J - self.cf_eval(R2 @ Wk) >= mu * mu_c[opt] * innGkHk:
                mu = 2 * mu
                R1 = R2
                R2 = R1 @ R1
            while J - self.cf_eval(R1 @ Wk) < (1/2) * mu * mu_c[opt] * innGkHk:
                mu = (1/2) * mu
                R1 = expm(mu * self.skew(Hk)) 
        return abs(mu)

    def geod_search_armijo2(self, Wk, Gk, Hk, opt):
        if opt not in ['min', 'max']:
            raise ValueError('`opt` must be either `max` or `min`')
        mu = 1
        if opt == 'min':
            R1 = expm(-mu * self.skew(Hk))
            R2 = R1 @ R1
            J = self.cf_eval(Wk)
            innGkHk = self.innerprod(Gk, Hk)
            if (J - self.cf_eval(R2 @ Wk) >= mu * innGkHk) and (J - self.cf_eval(R1 @ Wk) < (1/2) * mu * innGkHk):
                pass
            else:
                while J - self.cf_eval(R2 @ Wk) >= mu * innGkHk:
                    mu = 2 * mu
                    R1 = R2
                    R2 = R1 @ R1
                while J - self.cf_eval(R1 @ Wk) < (1/2) * mu * innGkHk:
                    mu = 1/2 * mu
                    R1 = expm(-mu * self.skew(Hk))
        elif opt == 'max':
            R1 = expm(mu * self.skew(Hk))
            R2 = R1 @ R1
            J = self.cf_eval(Wk)
            innGkHk = self.innerprod(Gk, Hk)
            if (J - self.cf_eval(R2 @ Wk) <= -mu * innGkHk) and (J - self.cf_eval(R1 @ Wk) > (1/2) * mu * innGkHk):
                pass
            else:
                while J - self.cf_eval(R2 @ Wk) <= -mu * innGkHk:
                    mu = 2 * mu
                    R1 = R2
                    R2 = R1 @ R1
                while J - self.cf_eval(R1 @ Wk) > -(1/2) * mu * innGkHk:
                    mu = 1/2 * mu
                    R1 = expm(mu * self.skew(Hk))
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
        R_mu_poly = np.eye(Wk.shape[0])
        for n_poly in range(N_poly+1):
            Dk = self.euclid_grad_eval(R_mu_poly @ Wk)
            d1_poly[n_poly] = - 2 * sign_mu * np.real(np.trace(Dk @ Wk.conjugate().T @ R_mu_poly.conjugate().T @ Hk.conjugate().T))
            R_mu_poly = R_mu_poly @ R_poly

        C = np.array([[] for _ in range(N_poly)])
        mu_poly = np.linspace(0, T_mu, N_poly+1)
        for n_poly in range(1, N_poly + 1):
            C = np.c_[C, mu_poly[1:]**n_poly]
        
        d_new = d1_poly[1:] - d1_poly[0]
        d_new[-1] = 0
        a = np.r_[d1_poly[0], inv(C) @ d_new]
        # a = np.r_[d1_poly[0], inv(C) @ (d1_poly[1:] - d1_poly[0])]
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
        R_mu_dft = np.eye(Wk.shape[0])
        J_dft = np.zeros(N_dft)
        d1_dft = np.zeros(N_dft)
        for n_dft in range(N_dft):
            Dk = self.euclid_grad_eval(R_mu_dft @ Wk)
            d1_dft[n_dft] = -2 * sign_mu * np.real(np.trace(Dk @ Wk.conjugate().T @ R_mu_dft.conjugate().T @ Hk.conjugate().T))
            J_dft[n_dft] = self.cf_eval(R_mu_dft @ Wk)
            R_mu_dft = R_mu_dft @ R_dft

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
        if mu_opt_dft == [] or abs(d1_dft[0]) <= 1.e-12:
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

    def cf_eval(self, Wk):
        return np.real(np.trace(Wk.conjugate().T @ self.S @ Wk @ self.N))

    def euclid_grad_eval(self, Wk):
        return self.S @ Wk @ self.N

    def renyi_ent(self, Wk):
        pass

    def renyi_ent_euclid_grad(self, Wk):
        pass


    _av_geod_search_methods = {'a': geod_search_armijo2, 'p': geod_search_poly, 'f': geod_search_dft} 

    _cost_functions = {'brockett': cf_eval, 'renyi': renyi_ent}
    _deriv_cost_functions = {'brockett': euclid_grad_eval, 'renyi': renyi_ent_euclid_grad}

    def __init__(self, W0, grad_method, geod_search_method, opt, K_iter, S, N, q):
        self.W0 = W0
        self.grad_method = grad_method
        self.geod_search_method = geod_search_method
        self.opt = opt
        self.K_iter = K_iter
        self.test_sanity()
        self.n = self.W0.shape[0]
        self.S = S
        self.N = N
        self.q = q
        if opt == 'min':
            self.sign_mu = -1
        elif opt == 'max':
            self.sign_mu = 1
        else:
            raise ValueError("Valid strings for parameter opt: `min` or `max`.")

    def test_sanity(self):
        if not isinstance(self.W0, np.ndarray):
            raise TypeError("W0 should be numpy array.")
        if self.W0.ndim != 2:
            raise TypeError("W0 has to be 2D matrix.")
        if self.W0.shape[0] != self.W0.shape[1]:
            raise TypeError("W0 must be square matrix")
        if norm(self.W0 @ self.W0.conjugate().T - np.eye(self.W0.shape[0])) >= 1.e-5:
            raise TypeError("W0 must be unitary.")

    def run_opt(self):

        Wk = self.W0.copy()
        J = np.zeros((self.K_iter, 1))
        E = np.zeros((self.K_iter, 1))
        U = np.zeros((self.K_iter, 1))
        
        for k in range(self.K_iter):
            J[k, 0] = self.cf_eval(Wk)
            E[k, 0] = self.diag_crit_eval(Wk)
            U[k, 0] = self.unit_crit_eval(Wk)
            if k % self.n**2 == 0:
                GEk = self.euclid_grad_eval(Wk)
                Gk = GEk @ Wk.conjugate().T - Wk @ GEk.conjugate().T
                Hk = Gk
            else:
                Gk = Gkplus1
                Hk = Hkplus1 

            # Choose geodesic (line) search method
            N_poly = 5
            NT = 5
            par2 = {'a': Gk, 'p': Hk, 'f': Hk}
            par3 = {'a': Hk, 'p': N_poly, 'f': NT}
            geod_method = self._av_geod_search_methods[self.geod_search_method].__get__(self, type(self))
            mu = geod_method(Wk, par2[self.geod_search_method], par3[self.geod_search_method], self.opt)
            R1 = expm(self.sign_mu * mu * self.skew(Hk))

            # Update
            Wkplus1 = R1 @ Wk
            GEkplus1 = self.euclid_grad_eval(Wkplus1)
            Gkplus1 = GEkplus1 @ Wkplus1.conjugate().T - Wkplus1 @ GEkplus1.conjugate().T
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

        return {"W_final": Wk, "J_dB": 10 * np.log10(J), "E_dB": 10 * np.log10(E), "U_dB": 10 * np.log10(U[1:])}
        # TODO make log10 to work with input 0
            
    
    def this_log10(self, stat):
        pass

    

    def diag_crit_eval(self, Wk):
        D = np.real(np.diag(np.diag(Wk.conjugate().T @ self.S @ Wk)))
        return norm(Wk.conjugate().T @ self.S @ Wk - D)
        # TODO finish this function (check if np.linalg.norm is the same as matlab norm)

    def unit_crit_eval(self, Wk):
        return norm(Wk.conjugate().T @ Wk - np.eye(self.n))**2

    def skew(self, Hk):
        return 0.5 * (Hk - Hk.conjugate().T)

    def innerprod(self, G1, G2):
        return 0.5 * np.real(np.trace(G1.conjugate().T @ G2))



