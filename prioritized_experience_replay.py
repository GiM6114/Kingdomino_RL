import torch
import random
import numpy as np

from tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, boards_state_size, cur_tiles_state_size, prev_tiles_state_size, action_size, buffer_size, eps=1e-2, alpha=0.1, beta=0.1, device='cpu'):
        self.device = device
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        # state: boards, current_tiles, previous_tiles       
        self.boards_state = torch.empty(buffer_size, *boards_state_size, dtype=torch.int, device=self.device)
        self.cur_tiles_state = torch.empty(buffer_size, cur_tiles_state_size, dtype=torch.int, device=self.device)
        self.prev_tiles_state = torch.empty(buffer_size, prev_tiles_state_size, dtype=torch.int, device=self.device)
        
        self.action = torch.empty(buffer_size, action_size, dtype=torch.int, device=self.device)
        self.reward = torch.empty(buffer_size, dtype=torch.float, device=self.device)
        
        self.boards_next_state = torch.empty(buffer_size, *boards_state_size, dtype=torch.int, device=self.device)
        self.cur_tiles_next_state = torch.empty(buffer_size, cur_tiles_state_size, dtype=torch.int, device=self.device)
        self.prev_tiles_next_state = torch.empty(buffer_size, prev_tiles_state_size, dtype=torch.int, device=self.device)
        
        self.done = torch.empty(buffer_size, dtype=torch.int, device=self.device)
        self.possible_actions = []

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
    
    def __len__(self):
        return self.real_size

    def add(self, transition):
        boards_state, cur_tiles_state, prev_tiles_state, \
            action, reward, \
            boards_next_state, cur_tiles_next_state, prev_tiles_next_state, \
            done, possible_actions = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.boards_state[self.count] = torch.as_tensor(boards_state, device=self.device)
        self.cur_tiles_state[self.count] = torch.as_tensor(cur_tiles_state, device=self.device)
        self.prev_tiles_state[self.count] = torch.as_tensor(prev_tiles_state, device=self.device)

        self.action[self.count] = torch.as_tensor(action, device=self.device)
        self.reward[self.count] = torch.as_tensor(reward, device=self.device)
        
        self.boards_next_state[self.count] = torch.as_tensor(boards_next_state, device=self.device)
        self.cur_tiles_next_state[self.count] = torch.as_tensor(cur_tiles_next_state, device=self.device)
        self.prev_tiles_next_state[self.count] = torch.as_tensor(prev_tiles_next_state, device=self.device)
        
        self.done[self.count] = torch.as_tensor(done, device=self.device)
        
        if len(self.possible_actions) == self.size:
            self.possible_actions[self.count] = torch.as_tensor(possible_actions, device=self.device)
        else:
            self.possible_actions.append(torch.as_tensor(possible_actions, device=self.device))

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = (
            self.boards_state[sample_idxs].to(self.device),
            self.cur_tiles_state[sample_idxs].to(self.device),
            self.prev_tiles_state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.boards_next_state[sample_idxs].to(self.device),
            self.cur_tiles_next_state[sample_idxs].to(self.device),
            self.prev_tiles_next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
            [self.possible_actions[i] for i in sample_idxs]
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class ReplayBuffer:
    def __init__(self, boards_state_size, cur_tiles_state_size, prev_tiles_state_size, action_size, buffer_size, device='cpu'):
        self.device = device
        # transition: state, action, reward, next_state, done
        # state: boards, current_tiles, previous_tiles       
        self.boards_state = torch.empty(buffer_size, *boards_state_size, dtype=torch.int, device=self.device)
        self.cur_tiles_state = torch.empty(buffer_size, cur_tiles_state_size, dtype=torch.int, device=self.device)
        self.prev_tiles_state = torch.empty(buffer_size, prev_tiles_state_size, dtype=torch.int, device=self.device)
        
        self.action = torch.empty(buffer_size, action_size, dtype=torch.int, device=self.device)
        self.reward = torch.empty(buffer_size, dtype=torch.float, device=self.device)
        
        self.boards_next_state = torch.empty(buffer_size, *boards_state_size, dtype=torch.int, device=self.device)
        self.cur_tiles_next_state = torch.empty(buffer_size, cur_tiles_state_size, dtype=torch.int, device=self.device)
        self.prev_tiles_next_state = torch.empty(buffer_size, prev_tiles_state_size, dtype=torch.int, device=self.device)
        
        self.done = torch.empty(buffer_size, dtype=torch.int, device=self.device)
        self.possible_actions = []

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        boards_state, cur_tiles_state, prev_tiles_state, \
        action, reward, \
        boards_next_state, cur_tiles_next_state, prev_tiles_next_state, \
        done, possible_actions = transition

        # store transition in the buffer
        self.boards_state[self.count] = torch.as_tensor(boards_state)
        self.cur_tiles_state[self.count] = torch.as_tensor(cur_tiles_state)
        self.prev_tiles_state[self.count] = torch.as_tensor(prev_tiles_state)

        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        
        self.boards_next_state[self.count] = torch.as_tensor(boards_next_state)
        self.cur_tiles_next_state[self.count] = torch.as_tensor(cur_tiles_next_state)
        self.prev_tiles_next_state[self.count] = torch.as_tensor(prev_tiles_next_state)
        
        self.done[self.count] = torch.as_tensor(done)
        
        if len(self.possible_actions) == self.size:
            self.possible_actions[self.count] = torch.as_tensor(possible_actions)
        else:
            self.possible_actions.append(torch.as_tensor(possible_actions))

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        batch = (
            self.boards_state[sample_idxs].to(self.device),
            self.cur_tiles_state[sample_idxs].to(self.device),
            self.prev_tiles_state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.boards_next_state[sample_idxs].to(self.device),
            self.cur_tiles_next_state[sample_idxs].to(self.device),
            self.prev_tiles_next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
            [self.possible_actions[i] for i in sample_idxs]
        )
        return batch
    
    def __len__(self):
        return self.real_size