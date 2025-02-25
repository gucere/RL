import numpy as np
import torch
class Memory():
    def __init__(self, max_size=100000):
        self.max_size=max_size
        self.transitions = np.empty(max_size, dtype=object)
        self.size = 0
        self.current_idx = 0

    def add_transition(self, transition):
        converted_transition = []
        for t in transition:
            if isinstance(t, (np.ndarray, list, float, int)):
                converted_transition.append(
                    torch.tensor(t, dtype=torch.float32, device="cuda")
                )
            else:
                converted_transition.append(t)
        
        self.transitions[self.current_idx] = tuple(converted_transition)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size
    
    def sample(self, batch=1):
        batch = min(batch, self.size)
        indices = np.random.choice(self.size, batch, replace=False)
        return self.transitions[indices]

    def get_all_transitions(self):
        return self.transitions[:self.size]