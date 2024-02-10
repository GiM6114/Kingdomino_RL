import math

class EpsilonScheduler:
    def __init__(self):
        pass
    def eps(self):
        pass



class EpsilonDecay:
    def __init__(self, eps_start, eps_end, eps_decay):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_steps = 0
        
    def eps(self):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.n_steps / self.eps_decay)
        self.n_steps += 1
        return eps

class EpsilonDecayRestart(EpsilonDecay):
    def __init__(self, eps_start, eps_end, eps_decay, eps_restart, eps_restart_threshold):
        super().__init__(eps_start, eps_end, eps_decay)
        self.eps_restart_threshold = eps_restart_threshold
        self.eps_restart = eps_restart
    def eps(self):
        eps = super().eps()
        if eps < self.eps_restart_threshold:
            self.n_steps = 0
            self.eps_start = self.eps_restart
        return eps
    
# one episode : 26 steps
# 100 episodes per iteration
# 20 iterations

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    e = EpsilonDecayRestart(0.9,0.01,5000,0.5,0.0175)
    a = np.zeros(50000)
    for i in range(a.shape[0]):
        a[i] = e.eps()
    
    plt.plot(a)