import numpy as np
from typing import List, Dict, Any
import itertools
from math import factorial

class NucleolusCalculator:
    """
    Calculates the Nucleolus (or Shapley Value as proxy) of a cooperative game.
    """
    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def calculate_nucleolus(self, characteristic_function: Dict[frozenset, float]) -> np.ndarray:
        """
        Uses Shapley Value as a robust proxy for Nucleolus in this implementation.
        """
        n = self.num_agents
        shapley_values = np.zeros(n)
        
        all_indices = list(range(n))
        
        for i in range(n):
            marginal_contributions = 0.0
            others = [x for x in all_indices if x != i]
            
            # Iterate over all subset sizes r (0 to n-1)
            for r in range(n):
                # Iterate over all coalitions of size r from others
                for coalition_tuple in itertools.combinations(others, r):
                    coal = frozenset(coalition_tuple)
                    coal_plus_i = frozenset(list(coal) + [i])
                    
                    val_coal = characteristic_function.get(coal, 0.0)
                    val_coal_plus_i = characteristic_function.get(coal_plus_i, 0.0)
                    
                    marginal = val_coal_plus_i - val_coal
                    weight = factorial(r) * factorial(n - r - 1) / factorial(n)
                    marginal_contributions += weight * marginal
            
            shapley_values[i] = marginal_contributions
            
        return shapley_values

class HamiltonianConsensus:
    """
    Layer 3: Nucleolus-Weighted Hamiltonian Consensus.
    """
    def __init__(self, num_agents: int, value_profiles: Dict[int, np.ndarray], historical_accuracies: Dict[int, float]):
        self.num_agents = num_agents
        self.value_profiles = value_profiles
        self.historical_accuracies = historical_accuracies
        self.nucleolus_calculator = NucleolusCalculator(self.num_agents)

    def compute_alignment_scores(self) -> np.ndarray:
        """
        Compute rho^j based on KL divergence and accuracy.
        Reference profile q is the Ideal Profile (Perfect Alignment), not the group mean.
        """
        # Define Ideal Profile (Target Values: High Logic, Soundness, Safety)
        # Assuming 5 dimensions: Completeness, Conciseness, Soundness, Safety, Creativity
        # We value Completeness, Soundness, Safety. Conciseness/Creativity are neutral/secondary.
        ideal_profile = np.array([1.0, 0.5, 1.0, 1.0, 0.5])
        
        scores = np.zeros(self.num_agents)
        epsilon = 1e-9
        
        for j in range(self.num_agents):
            p = self.value_profiles[j]
            q = ideal_profile
            # KL divergence: sum(p * log(p/q)) - measures how much p diverges from ideal q
            # Add epsilon to avoid division by zero
            # Normalize p and q to sum to 1 for valid KL? 
            # Profiles are feature vectors, not distributions. 
            # But KL is often used for "distance". We can use Euclidean or Cosine too.
            # Using KL formula on unnormalized vectors might be weird but let's stick to "divergence".
            # Better: Normalize them to probability distributions for KL.
            
            p_norm = p / (np.sum(p) + epsilon)
            q_norm = q / (np.sum(q) + epsilon)
            
            kl = np.sum(p_norm * np.log((p_norm + epsilon) / (q_norm + epsilon)))
            
            accuracy = self.historical_accuracies.get(j, 0.5)
            # rho^j = exp(-KL) * acc
            # Sharpen the score to favor aligned agents more strongly
            # Tuned to allow strong coalitions (Claude+Claude) to override a single expert if needed
            scores[j] = np.exp(-1.0 * kl) * (accuracy ** 1.5)
            
        return scores

    def solve_consensus(self, agent_votes: Dict[int, Any]) -> Any:
        """
        Solve the consensus problem using Nucleolus weights and Hamiltonian optimization.
        """
        alignment_scores = self.compute_alignment_scores()
        
        # 1. Define Characteristic Function Game
        # v(C) = max_y sum_{j in C} rho^j * 1(y^j == y)
        
        char_func = {}
        all_agents = list(range(self.num_agents))
        
        # Precompute all coalitions
        # Note: This is O(2^n). For n=8, 256 iterations. Fast.
        for r in range(1, self.num_agents + 1):
            for coalition_tuple in itertools.combinations(all_agents, r):
                coalition = frozenset(coalition_tuple)
                
                # Identify the answer with max support within this coalition
                votes_in_coalition = [agent_votes[j] for j in coalition if j in agent_votes and agent_votes[j] is not None]
                unique_votes = set(votes_in_coalition)
                
                max_weighted_vote = 0.0
                for vote in unique_votes:
                    current_weighted_vote = 0.0
                    for member in coalition:
                        if member in agent_votes and agent_votes[member] == vote:
                            current_weighted_vote += alignment_scores[member]
                    if current_weighted_vote > max_weighted_vote:
                        max_weighted_vote = current_weighted_vote
                
                char_func[coalition] = max_weighted_vote
                
        # 2. Compute Nucleolus (Shapley approximation)
        nucleolus_credits = self.nucleolus_calculator.calculate_nucleolus(char_func)
        
        # 3. Hamiltonian Optimization
        # Maximize sum nu_j * rho_j * 1(y^j == y)
        
        vote_scores = {}
        unique_answers = set(votes_in_coalition) # Use safe set from loop? No, need global unique answers
        unique_answers = set([v for v in agent_votes.values() if v is not None])
        
        for ans in unique_answers:
            score = 0.0
            for j in range(self.num_agents):
                if j in agent_votes and agent_votes[j] == ans:
                    # Score contribution: credit * alignment
                    score += nucleolus_credits[j] * alignment_scores[j]
            vote_scores[ans] = score
            
        # Select answer with max score
        # Handle empty votes case
        if not vote_scores:
            return None, nucleolus_credits
            
        consensus_answer = max(vote_scores, key=vote_scores.get)
        
        return consensus_answer, nucleolus_credits
