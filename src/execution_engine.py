import numpy as np
import cvxpy as cp

class AlmgrenChrissOptimizer:
    def __init__(self, X, N, sigma, gamma, eta, tau):
        self.X = X          # Total shares to sell
        self.N = N          # Number of steps
        self.sigma = sigma  # Daily Volatility
        self.gamma = gamma  # Permanent Impact Coeff
        self.eta = eta      # Temporary Impact Coeff
        self.tau = tau      # Time step (e.g., 1/N)

    def solve(self, risk_aversion=1e-6):
        """Finds the optimal trade list using Quadratic Programming."""
        n = cp.Variable(self.N)
        # Inventory at each step
        x = self.X - cp.cumsum(cp.vstack([0, cp.reshape(n[:-1], (self.N-1, 1))]))
        
        # Objective: E[Cost] + lambda * Var[Cost]
        # E[Cost] approx sum(eta/tau * n^2 + 0.5 * gamma * n^2)
        expected_cost = cp.sum(self.eta/self.tau * cp.square(n) + 0.5 * self.gamma * cp.square(n))
        variance = cp.sum(cp.square(x)) * (self.sigma**2 * self.tau)
        
        obj = cp.Minimize(expected_cost + risk_aversion * variance)
        constraints = [cp.sum(n) == self.X, n >= 0]
        
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return n.value

    def simulate_path(self, n_list, S0):
        """Simulates a single price path given a trade list."""
        N = len(n_list)
        S = np.zeros(N + 1)
        S_exec = np.zeros(N)
        S[0] = S0
        
        for k in range(N):
            # Price process with permanent impact
            drift = -self.gamma * n_list[k]
            diffusion = self.sigma * np.sqrt(self.tau) * np.random.normal()
            S[k+1] = S[k] + diffusion + drift
            # Execution price with temporary impact
            S_exec[k] = S[k+1] - self.eta * (n_list[k] / self.tau)
            
        return S, S_exec
