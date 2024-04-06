def get_n_actions(board_size):
    return 4*board_size**2 - 4*board_size - 8

def action2id2action(board_size, n_players):
    middle = board_size // 2
    action2id = {}
    id2action = []
    k = 0
    n_actions = get_n_actions(board_size)
    for i in range(board_size):
        for j in range(board_size):
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if i+ii < 0 or i+ii > board_size-1 or j+jj < 0 or j+jj > board_size - 1:
                        continue
                    if (abs(ii) == abs(jj)):
                        continue
                    if (i == middle and j == middle) or (i+ii == middle and j+jj == middle):
                        continue
                    pos = ((i,j),(i+ii,j+jj))
                    for p in range(n_players):
                        action2id[(p,pos)] = k + p*n_actions
                        id2action.append((p,pos))
                    k += 1
    assert(k == n_actions), print(k,n_actions)
    assert (len(action2id.keys()) == n_players*n_actions)
    assert(len(id2action) == n_players*n_actions)
    return action2id,id2action

