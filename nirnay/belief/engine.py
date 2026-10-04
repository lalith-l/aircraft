"""
Belief Engine — Hidden Markov Model (HMM) for health state tracking.

Implements Forward-Backward smoothing and Expectation-Maximisation (EM)
parameter learning.

Status: [REAL]
"""

import math
import numpy as np
from typing import List, Tuple, Dict
from nirnay.contracts import EvidenceItem, HealthState

class BeliefEngine:
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        
        # Initial state distribution: usually starts in NEW (0) or GOOD (1)
        self.pi = np.array([0.8, 0.2, 0.0, 0.0, 0.0])
        
        # Transition matrix A[i, j] = P(S_t = j | S_{t-1} = i)
        # Initialize as upper-triangular (parts degrade, they don't self-heal)
        self.A = np.zeros((n_states, n_states))
        for i in range(n_states):
            # Probability of staying in state i
            self.A[i, i] = 0.95
            # Probability of degrading to worse states
            remaining = 1.0 - self.A[i, i]
            n_worse = n_states - i - 1
            if n_worse > 0:
                for j in range(i + 1, n_states):
                    self.A[i, j] = remaining / n_worse
        
        # Ensure row sums are 1.0 (state 4 is absorbing)
        self.A[n_states - 1, n_states - 1] = 1.0

    def _get_likelihoods(self, sequence: List[EvidenceItem]) -> np.ndarray:
        """Extract the emission likelihoods from a sequence of evidence items."""
        L = np.zeros((len(sequence), self.n_states))
        for t, item in enumerate(sequence):
            if "per_state" in item.likelihood:
                L[t, :] = item.likelihood["per_state"]
            else:
                # If no per_state likelihood (e.g. some manual entry), default to uniform
                L[t, :] = 1.0 / self.n_states
            # Weight could optionally exponentiate or scale the likelihood, 
            # but for now we assume weight is handled upstream or just use it directly.
            # Avoid absolute zeros
            L[t, :] = np.clip(L[t, :], 1e-12, 1.0)
        return L

    def forward_backward(self, sequence: List[EvidenceItem]) -> Tuple[np.ndarray, float]:
        """
        Run Forward-Backward smoothing on a sequence.
        
        Returns:
            gamma: (T, n_states) array of smoothed marginal probabilities P(S_t | O_{1:T})
            log_likelihood: float, log P(O_{1:T})
        """
        if not sequence:
            return np.zeros((0, self.n_states)), 0.0

        L = self._get_likelihoods(sequence)
        T = len(sequence)
        
        # Forward pass
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T) # Scaling factors
        
        # t = 0
        alpha[0] = self.pi * L[0]
        c[0] = np.sum(alpha[0])
        if c[0] == 0: c[0] = 1e-12
        alpha[0] /= c[0]
        
        # t > 0
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * L[t]
            c[t] = np.sum(alpha[t])
            if c[t] == 0: c[t] = 1e-12
            alpha[t] /= c[t]
            
        log_likelihood = np.sum(np.log(c))
        
        # Backward pass
        beta = np.zeros((T, self.n_states))
        beta[T-1] = 1.0 # beta is scaled by c[t] implicitly in standard scaled FB
        
        for t in range(T-2, -1, -1):
            beta[t] = (self.A @ (L[t+1] * beta[t+1])) / c[t+1]
            
        # Smoothed marginals (gamma)
        gamma = alpha * beta
        gamma_sums = np.sum(gamma, axis=1, keepdims=True)
        gamma_sums[gamma_sums == 0] = 1e-12
        gamma /= gamma_sums
        
        return gamma, log_likelihood

    def _e_step(self, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """E-step for a single sequence (returns expected counts)."""
        T = len(L)
        
        # Forward
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)
        
        alpha[0] = self.pi * L[0]
        c[0] = np.sum(alpha[0])
        if c[0] == 0: c[0] = 1e-12
        alpha[0] /= c[0]
        
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * L[t]
            c[t] = np.sum(alpha[t])
            if c[t] == 0: c[t] = 1e-12
            alpha[t] /= c[t]
            
        log_likelihood = np.sum(np.log(c))
        
        # Backward
        beta = np.zeros((T, self.n_states))
        beta[T-1] = 1.0
        
        for t in range(T-2, -1, -1):
            beta[t] = (self.A @ (L[t+1] * beta[t+1])) / c[t+1]
            
        # Gamma
        gamma = alpha * beta
        gamma_sums = np.sum(gamma, axis=1, keepdims=True)
        gamma_sums[gamma_sums == 0] = 1e-12
        gamma /= gamma_sums
        
        # Xi (two-slice marginals)
        xi_sum = np.zeros((self.n_states, self.n_states))
        for t in range(1, T):
            # xi_t = alpha_{t-1} * A * L_t * beta_t
            xi_t = (alpha[t-1][:, None] * self.A) * (L[t] * beta[t])[None, :]
            xi_sum += xi_t / c[t]
            
        return gamma, xi_sum, log_likelihood

    def learn_em(self, sequences: List[List[EvidenceItem]], max_iter: int = 50, tol: float = 1e-4) -> List[float]:
        """
        Run Baum-Welch EM to learn the transition matrix A and initial dist pi.
        
        Args:
            sequences: List of EvidenceItem sequences (e.g., one per part SN).
            max_iter: Maximum number of EM iterations.
            tol: Convergence tolerance for log-likelihood.
            
        Returns:
            log_likelihoods: List of total log-likelihoods per iteration.
        """
        # Convert sequences to likelihood arrays once
        L_seqs = [self._get_likelihoods(seq) for seq in sequences if len(seq) > 1]
        
        if not L_seqs:
            return []
            
        log_likelihoods = []
        
        for iteration in range(max_iter):
            total_ll = 0.0
            
            gamma_0_sum = np.zeros(self.n_states)
            xi_sum_total = np.zeros((self.n_states, self.n_states))
            gamma_sum_total = np.zeros(self.n_states)
            
            for L in L_seqs:
                gamma, xi_sum, ll = self._e_step(L)
                total_ll += ll
                
                gamma_0_sum += gamma[0]
                xi_sum_total += xi_sum
                # We sum gamma from t=0 to T-2 for the transition denominator
                gamma_sum_total += np.sum(gamma[:-1], axis=0)
                
            log_likelihoods.append(total_ll)
            
            # M-step
            # Update pi
            self.pi = gamma_0_sum / np.sum(gamma_0_sum)
            
            # Update A
            for i in range(self.n_states):
                if gamma_sum_total[i] > 1e-12:
                    self.A[i, :] = xi_sum_total[i, :] / gamma_sum_total[i]
                else:
                    # Keep as is if no data
                    pass
            
            # Enforce upper triangular structure (no self-healing)
            # by zeroing out lower triangle and re-normalising
            self.A = np.triu(self.A)
            
            row_sums = np.sum(self.A, axis=1, keepdims=True)
            # Avoid div by zero
            row_sums[row_sums == 0] = 1.0
            self.A /= row_sums
            
            # Check convergence
            if iteration > 0 and (total_ll - log_likelihoods[-2]) < tol:
                break
                
        return log_likelihoods
