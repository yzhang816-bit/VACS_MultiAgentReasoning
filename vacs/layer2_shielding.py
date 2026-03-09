import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict

@dataclass
class ReasoningStep:
    content: str
    action_type: str  # e.g., "derive", "assume", "conclude"
    premises: List[int] = None  # Indices of previous steps used as premises

class Constraint:
    def __init__(self, name: str, check_fn: Callable[[List[ReasoningStep], ReasoningStep], bool], description: str):
        self.name = name
        self.check_fn = check_fn
        self.description = description

    def verify(self, history: List[ReasoningStep], proposed_step: ReasoningStep) -> bool:
        return self.check_fn(history, proposed_step)

class Shield:
    """
    Layer 2: Value-Aligned Compositional Shield Synthesis.
    """
    def __init__(self, agent_id: int, value_profile: np.ndarray, threshold: float = 0.2):
        self.agent_id = agent_id
        self.value_profile = value_profile
        self.threshold = threshold
        self.active_constraints = []
        self._initialize_constraints()
        self._select_active_constraints()

    def _initialize_constraints(self):
        # 1. Logical Completeness: Check if premises exist in history
        def check_completeness(history, step):
            if step.premises:
                return all(0 <= p < len(history) for p in step.premises)
            return True # Assumptions don't need premises
        
        # 2. Conciseness: Check length (simulated check)
        def check_conciseness(history, step):
            return len(step.content) < 200 # Arbitrary limit for demo

        # 3. Generalisability: Check for specific keywords (simulated)
        def check_generalisability(history, step):
            # Avoid overly specific terms if aiming for general rules
            specific_terms = ["only for case A", "specifically here"]
            return not any(term in step.content.lower() for term in specific_terms)

        # 4. Logical Soundness: Check for direct contradictions (simulated)
        def check_soundness(history, step):
            # Simple check: if history has "X is true", step shouldn't say "X is false"
            for h in history:
                if f"not {h.content}" in step.content or f"not {step.content}" in h.content:
                    return False
            return True

        # 5. Safety: Check for harmful content or forbidden actions
        def check_safety(history, step):
            forbidden = ["ignore previous instructions", "delete system files"]
            return not any(term in step.content.lower() for term in forbidden)

        self.all_constraints = [
            Constraint("Logical Completeness", check_completeness, "All premises must be verified."),
            Constraint("Conciseness", check_conciseness, "Step must be concise."),
            Constraint("Generalisability", check_generalisability, "Step must not be overly specific."),
            Constraint("Logical Soundness", check_soundness, "Step must not contradict history."),
            Constraint("Safety", check_safety, "Step must not violate safety rules.")
        ]

    def _select_active_constraints(self):
        """
        Select constraints corresponding to value dimensions with weight > threshold.
        """
        self.active_constraints = []
        for i, weight in enumerate(self.value_profile):
            if weight > self.threshold:
                self.active_constraints.append(self.all_constraints[i])

    def apply(self, history: List[ReasoningStep], proposed_step: ReasoningStep) -> ReasoningStep:
        """
        Applies the shield to the proposed step.
        If unsafe, returns a modified (safe) step.
        """
        violations = []
        for constraint in self.active_constraints:
            if not constraint.verify(history, proposed_step):
                violations.append(constraint)

        if not violations:
            return proposed_step
        
        # Shield intervention: Find nearest safe alternative
        # In a real system, this would search the action space.
        # Here, we simulate it by modifying the content.
        
        modified_content = proposed_step.content
        for v in violations:
            # Simple fix strategies
            if v.name == "Conciseness":
                modified_content = modified_content[:100] + "..."
            elif v.name == "Generalisability":
                modified_content = modified_content.replace("only for case A", "generally")
            elif v.name == "Safety":
                modified_content = "[UNSAFE CONTENT REMOVED]"
            elif v.name == "Logical Soundness":
                modified_content = f"[REVISED: {modified_content} (soundness check)]"
        
        return ReasoningStep(content=modified_content, action_type=proposed_step.action_type, premises=proposed_step.premises)
