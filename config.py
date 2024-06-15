from kingdomino.rewards import player_focused_reward
from networks import PlayerFocusedFC, ConvShared 

action_types = ['singular', 'sequential']
network_types = [PlayerFocusedFC, ConvShared]
reward_fns = [player_focused_reward]
opponents = ['self', 'random']

network_hp = {
  'player_encoder':{
      'board_encoder':{
            'cnn':[
                {'type':'conv', 'params':{'out_channels':16, 'kernel_size':5, 'stride':1, 'padding':2}},
                {'type':'maxpool','params':{'kernel_size':2, 'stride':1, 'padding':1}},
                {'type':'conv', 'params':{'out_channels':16, 'kernel_size':5, 'stride':1, 'padding':2}},
                {'type':'maxpool','params':{'kernel_size':2, 'stride':1, 'padding':1}},
                {'type':'conv', 'params':{'out_channels':16, 'kernel_size':5, 'stride':1, 'padding':2}},
                {'type':'maxpool','params':{'kernel_size':2, 'stride':1, 'padding':1}},
                ],
            'fc':{
                'output_size':128,
                'l':3,
                'n':128}},
      'prev_tile_encoder':{
          'output_size':64,
          'l':2,
          'n':64},
      'joint_encoder':{
          'output_size':256,
          'l':3,
          'n':128}},
  'cur_tiles_encoder':{
      'output_size':64,
      'l':2,
      'n':64},
  'joint_encoder':{
      'l':4,
      'n':256}
      }

hp = {'batch_size':256,
      'tau':0.005,
      'gamma':0.99,
      'lr':3e-4,
      'replay_memory_size':50000,
      'double':True,
      'PER':True,
      'reward_name_id':0,
      'opponent_type':'random',
      # Exploration
      'eps_start':0.9,
      'eps_end':0.01,
      'eps_decay':1000,
      'eps_restart_threshold':0.0175,
      'eps_restart':0.5,
      # Architecture
      'network_type':1,
      'network_hp':network_hp,
      'action_type':'sequential',
      }

experiment = {
    'hp':hp,
    'board_size':5,
    'n_players':2,}