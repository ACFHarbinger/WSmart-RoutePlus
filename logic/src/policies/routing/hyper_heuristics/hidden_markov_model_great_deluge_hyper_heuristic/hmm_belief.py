"""Observation alphabet and hidden-state belief updates for the HMM-GD hyper-heuristic, mixed into HMMGDHHSolver as HMMBeliefMixin."""

import numpy as np

# Discrete Observation Alphabet (Onsem et al. 2014, Section 5.1)
_OBS_IMPROVE = 0  # ΔProfit > 0
_OBS_WORSE = 1  # ΔProfit < 0, ΔCost = 0
_OBS_WORSE_COST = 2  # Fix #R-Final-1 (Semantic Renaming)
_OBS_SAME = 3  # ΔProfit = 0, ΔCost = 0
_OBS_SAME_COST = 4  # Fix #R-Final-1 (Semantic Renaming)
_N_OBS = 5


class HMMBeliefMixin:
    # ------------------------------------------------------------------
    # HHaaHMM Logic
    # ------------------------------------------------------------------

    def _map_observation(self, delta_profit: float, delta_cost: float) -> int:
        """Map profit/cost changes to discrete alphabet (Fix #R-Final-1).

        Args:
            delta_profit: Change in profit.
            delta_cost: Change in cost.

        Returns:
            int: Index of the observation.
        """
        if delta_profit > 1e-9:
            return _OBS_IMPROVE
        if delta_profit < -1e-9:
            return _OBS_WORSE_COST if delta_cost > 1e-9 else _OBS_WORSE
        # delta_profit is roughly 0
        return _OBS_SAME_COST if delta_cost > 1e-9 else _OBS_SAME

    def _check_state_scaling(self) -> None:
        """Scale number of states with O(sqrt(log t)) (Fix #3).

        Returns:
            None: Modifies the state of the solver.
        """
        target_n = int(np.ceil(np.sqrt(np.log(self.t + 1)))) + 1
        if target_n > self.n_states:
            self._split_state()

    def _split_state(self) -> None:
        """
        Split the "least determined" state (Fix #1: Combined Entropy Split).

        Following Onsem et al. (2014) Section 6, the state with the highest combined
        uncertainty in both output character and next state is chosen for splitting.

        Returns:
            None: Modifies the state of the solver.
        """
        uncertainties = []
        for s in range(self.n_states):
            # Emission entropy H(B|s) summed over LLHs
            h_b = -np.sum(self._B[:, s, :] * np.log(self._B[:, s, :] + 1e-12))
            # Transition entropy H(A|s) summed over LLHs
            h_a = -np.sum(self._A[:, s, :] * np.log(self._A[:, s, :] + 1e-12))
            uncertainties.append(h_b + h_a)

        s_split = int(np.argmax(uncertainties))

        # Increment n_states
        old_n = self.n_states
        new_n = old_n + 1
        self.n_states = new_n

        # Resize Matrices (A is [u, s, s'], B is [u, s, o])
        new_A = np.ones((self.n_llh, new_n, new_n)) / new_n
        new_B = np.ones((self.n_llh, new_n, _N_OBS)) / _N_OBS

        # Copy old values
        new_A[:, :old_n, :old_n] = self._A
        new_B[:, :old_n, :] = self._B

        # --- Refinement 2: Transition Matrix Inheritance (Fix #R2) ---
        # The new state inherits outgoing transitions from s_split
        new_A[:, new_n - 1, :old_n] = self._A[:, s_split, :]
        # The new state inherits incoming transitions to s_split
        new_A[:, :old_n, new_n - 1] = self._A[:, :, s_split]
        # Self-transition for the new state inherits from parent
        new_A[:, new_n - 1, new_n - 1] = self._A[:, s_split, s_split]

        # Apply small random perturbations to break symmetry (Transitions)
        perturb_a = np.array(
            [
                [[self.random.uniform(-0.01, 0.01) for _ in range(new_n)] for _ in range(new_n)]
                for _ in range(self.n_llh)
            ]
        )
        new_A[:, new_n - 1, :] += perturb_a[:, new_n - 1, :]
        new_A = np.clip(new_A, 1e-12, 1.0)
        new_A /= new_A.sum(axis=2, keepdims=True)

        # Apply small random perturbations (Emissions)
        perturb_b = np.array([[self.random.uniform(-0.01, 0.01) for _ in range(_N_OBS)] for _ in range(self.n_llh)])
        new_B[:, new_n - 1, :] = self._B[:, s_split, :] + perturb_b
        new_B = np.clip(new_B, 1e-12, 1.0)
        new_B /= new_B.sum(axis=2, keepdims=True)

        self._A = new_A
        self._B = new_B
        self._counts_A = np.zeros((self.n_llh, new_n, new_n))
        self._counts_B = np.zeros((self.n_llh, new_n, _N_OBS))

        # --- Resize State-Aware Profit Matrices (Fix #R1) ---
        new_expected_profit = np.zeros((new_n, self.n_llh))
        new_profit_counts = np.zeros((new_n, self.n_llh))
        new_expected_profit[:old_n, :] = self._expected_profit
        new_profit_counts[:old_n, :] = self._profit_counts
        # New state inherits profit expectations from s_split
        new_expected_profit[new_n - 1, :] = self._expected_profit[s_split, :]
        # Split history counts
        new_profit_counts[new_n - 1, :] = self._profit_counts[s_split, :] / 2.0
        self._profit_counts[s_split, :] /= 2.0
        self._expected_profit = new_expected_profit
        self._profit_counts = new_profit_counts

        # Resize belief
        new_belief = np.zeros(new_n)
        new_belief[:old_n] = self._belief * 0.5
        new_belief[s_split] = self._belief[s_split] * 0.5
        new_belief[new_n - 1] = self._belief[s_split]  # shared belief
        self._belief = new_belief / new_belief.sum()

    def _online_em_update(self, u_idx: int, o_idx: int):
        """
        Stochastic Online EM approximation for HHaaHMM (Fix #R-Final-3).

        Due to the continuous streaming nature of the hyper-heuristic, this replaces
        the batch offline Baum-Welch algorithm with a Stochastic Online EM
        approximation. It estimates the joint transition probabilities using
        strictly the Forward variables, applying a decay factor γ=0.9 to
        construct an exponential moving average of the state-transition counts.

        Args:
            u_idx: Index of the Low-Level Heuristic (LLH).
            o_idx: Index of the observation.
        """
        # Forward Step: alpha_t(j) = P(o_t | s_j, u_t) * sum_i alpha_{t-1}(i) * P(s_j | s_i, u_t)
        new_belief = np.zeros(self.n_states)
        for j in range(self.n_states):
            transmission = np.dot(self._belief, self._A[u_idx, :, j])
            new_belief[j] = self._B[u_idx, j, o_idx] * transmission

        belief_sum = new_belief.sum()
        if belief_sum > 1e-12:
            new_belief /= belief_sum
        else:
            new_belief = np.ones(self.n_states) / self.n_states

        # Update counts for online EM (simplified online updates)
        # counts_A(i,j) += P(s_t=i, s_{t+1}=j | history)
        # counts_B(i,o) += P(s_t=i | history and obs o)
        for i in range(self.n_states):
            for j in range(self.n_states):
                # Approximation of joint probability for online use
                joint = self._belief[i] * self._A[u_idx, i, j] * self._B[u_idx, j, o_idx]
                self._counts_A[u_idx, i, j] = self.gamma * self._counts_A[u_idx, i, j] + joint

            self._counts_B[u_idx, i, o_idx] = self.gamma * self._counts_B[u_idx, i, o_idx] + self._belief[i]

        # Re-normalise A and B
        for i in range(self.n_states):
            # Normalise A[u, i, :]
            row_sum_a = self._counts_A[u_idx, i, :].sum()
            if row_sum_a > 1e-12:
                self._A[u_idx, i, :] = self._counts_A[u_idx, i, :] / row_sum_a

            # Normalise B[u, i, :]
            row_sum_b = self._counts_B[u_idx, i, :].sum()
            if row_sum_b > 1e-12:
                self._B[u_idx, i, :] = self._counts_B[u_idx, i, :] / row_sum_b

        self._belief = new_belief

    def _select_action(self, iteration: int, max_iterations: int) -> int:
        """
        Entropy-Maximizing Action Selection with Dynamic Annealing (Fix #5, #R1, #R-Final-1 & #Final-3).

        score(u) = NormalizedProfit(u) + current_alpha * Entropy(P(s_{t+1}|u))

        Args:
            iteration: Current iteration.
            max_iterations: Maximum number of iterations.

        Returns:
            int: Index of the selected action.
        """
        # 1. Calculate raw expected profits and future entropies
        raw_profits = np.zeros(self.n_llh)
        entropies = np.zeros(self.n_llh)
        for u in range(self.n_llh):
            # Predicted next state distribution for action u: P(s' | u) = sum_i P(s_i|hist) * P(s'|s_i,u)
            p_next = np.dot(self._belief, self._A[u, :, :])
            # Entropy: force distribution towards uniform to explore
            entropies[u] = -np.sum(p_next * np.log(p_next + 1e-12))

            # --- Refinement 1: State-Aware Profit Expectation (Fix #R1) ---
            # E[Profit | u] = sum_s P(s | history) * E[Profit | s, u]
            raw_profits[u] = np.dot(self._belief, self._expected_profit[:, u])

        # 2. Robust Min-Max Normalization (Fix #R-Final-1 & #3: Neutralize Ties)
        # Binds profits to [0, 1] so they are comparable to entropy (bounded by ln(n)).
        # If all profits are equal (delta < 1e-9), set to 0.5 to allow entropy to drive exploration.
        p_min, p_max = raw_profits.min(), raw_profits.max()
        if p_max - p_min > 1e-9:
            normalized_profits = (raw_profits - p_min) / (p_max - p_min)
        else:
            normalized_profits = np.full(self.n_llh, 0.5)

        # 3. Dynamic Exploration Annealing (Fix #R-Final-3)
        # alpha decays linearly to zero to transition from exploration to deep exploitation
        current_alpha = self.alpha * (1.0 - (iteration / max_iterations))

        # 4. Combine normalized profit and annealed entropy exploration bonus
        scores = normalized_profits + current_alpha * entropies

        return int(np.argmax(scores))
