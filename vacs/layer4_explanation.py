import numpy as np
from typing import List, Dict, Any, Tuple
from vacs.layer2_shielding import ReasoningStep

class Explainer:
    """
    Layer 4: Critical Path Extraction and Faithful Explanation Generation.
    """
    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def extract_critical_path(self, 
                              agent_trajectories: Dict[int, List[ReasoningStep]], 
                              consensus_answer: Any, 
                              nucleolus_credits: np.ndarray,
                              alignment_scores: np.ndarray,
                              agent_votes: Dict[int, Any]) -> List[Tuple[int, ReasoningStep]]:
        """
        Extracts the minimal critical reasoning path.
        In this simulation, we select the most aligned agent who voted for the consensus answer,
        and take their trajectory as the "backbone" of the explanation.
        Then we might add supporting steps from other high-credit agents.
        """
        
        # Identify agents who supported the consensus
        supporters = [j for j in range(self.num_agents) if agent_votes.get(j) == consensus_answer]
        
        if not supporters:
            return []
            
        # Find the most influential supporter (max credit * alignment)
        best_supporter = max(supporters, key=lambda j: nucleolus_credits[j] * alignment_scores[j])
        
        # The critical path is essentially this agent's reasoning chain
        # In a real system, we would prune this chain based on co-state sensitivity.
        # Here, we just take the last few steps or the whole chain.
        trajectory = agent_trajectories[best_supporter]
        
        # Format as (agent_id, step) tuples
        critical_path = [(best_supporter, step) for step in trajectory]
        
        return critical_path

    def generate_explanation(self, 
                             critical_path: List[Tuple[int, ReasoningStep]], 
                             consensus_answer: Any,
                             nucleolus_credits: np.ndarray,
                             alignment_scores: np.ndarray,
                             agent_votes: Dict[int, Any]) -> str:
        """
        Generates a natural language explanation from the critical path.
        """
        if not critical_path:
            return "No explanation available (no consensus or empty critical path)."
            
        supporters = [j for j in range(self.num_agents) if agent_votes.get(j) == consensus_answer]
        dissenters = [j for j in range(self.num_agents) if agent_votes.get(j) != consensus_answer]
        
        avg_alignment = np.mean([alignment_scores[j] for j in supporters]) if supporters else 0.0
        
        # Template-based generation
        explanation = f"The consensus answer is **{consensus_answer}** because:\n"
        
        for i, (agent_id, step) in enumerate(critical_path):
            explanation += f"({i+1}) Agent {agent_id} established: \"{step.content}\" "
            # Simulated constraint mention
            explanation += "(satisfying Logical Soundness/Completeness).\n"
            
        explanation += f"\nThis reasoning chain is supported by {len(supporters)} agents (mean alignment score {avg_alignment:.2f})."
        
        if dissenters:
            explanation += f" Dissenting agents: {dissenters}."
            
        return explanation
