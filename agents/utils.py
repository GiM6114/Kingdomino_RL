import torch

def listdict2dicttensor(state_batch):
    state_batch_dict = {key:value for key,value in state_batch[0].items()}
    for state in state_batch[1:]:
        for key in state_batch_dict.keys():
            state_batch_dict[key] = torch.cat((state_batch_dict[key], state[key]), dim=0)
    return state_batch_dict