import numpy as np

from utils import arr_except

def get_delta_current_best(current, scores):
    best_id = np.argmax(arr_except(scores, except_id=current))
    if best_id >= current:
        best_id += 1
    return scores[current] - scores[best_id]
     

def reward_last_quantitative(kd):
    if kd.first_turn:
        return 0
    if not kd.empty_end_turn and not (kd.last_turn and kd.current_player_itr == kd.n_players-1):
        return 0
    scores = kd.scores()
    return get_delta_current_best(kd.current_player_id, scores)/20

# 1 or -1
def reward_last_qualitative(kd, terminated):
    pass

# difference between scores
def reward_each_turn(kd):
    if kd.first_turn:
        return 0
    scores = kd.scores()
    return get_delta_current_best(kd.current_player_id, scores)

# difference between difference of previous and current scores
# should enable better credit assignment
def reward_delta_each_turn(kd):
    scores = kd.scores()
    if kd.first_turn:
        kd.prev_scores = np.zeros((kd.n_players,kd.n_players))
        kd.prev_scores[kd.current_player_id] = scores
        return 0
    reward = get_delta_current_best(kd.current_player_id, scores) - \
        get_delta_current_best(kd.current_player_id, kd.prev_scores[kd.current_player_id])
    kd.prev_scores[kd.current_player_id] = scores
    return reward

def player_focused_reward(kd, p_id):
    scores = kd.getScores()
    if kd.first_turn:
        return 0
    reward = scores[p_id] - kd.prev_scores[p_id]
    return reward

def reward_score(kd, p_id):
    return kd.getScore(p_id)