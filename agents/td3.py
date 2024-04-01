import torch

from agents.base import LearningAgent

class TD3(LearningAgent):
    def __init__(self,
                 memory,
                 noise_std,):
        pass
    
    def select_action(self, state, kingdomino):
        action = self.policy(state)
        if self.training:
            action = torch.clip(
                 action + torch.normal(0,self.noise_std),self.action_low,self.action_high)
        return action
        
        
    def processReward(self, reward, next_state, done):
        pass