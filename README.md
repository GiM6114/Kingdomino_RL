### Introduction

This repository is an implementation of the board game Kingdomino, on which I intend to apply deep reinforcement learning techniques as a way to challenge myself (which is why I try to not use pre-existing RL libraries like stable-baselines3) and perhaps find interesting new strategies for the game.
A video explaining the rules of the game can be found here: https://www.youtube.com/watch?v=smbwBPmP4Ms&ab_channel=WatchItPlayed.
Some of the questions I wish to answer is whether it is a better strategy to mess with the plans of other players rather than maximizing one's own score, whether player should focus more often on picking small scores tiles as a way to play first more often, etc.
Previous work on Kingdomino playing agents using Monte-Carlo methods, done by Gedda M. et al: https://arxiv.org/abs/1807.04458.

### Reasons for Kingdomino

Here is a couple reasons why Kingdomino is an interesting and challenging game:

- Multimodal data: 2D data (boards) as well as tabular data (tiles)
- Multiplayer (although the interaction between players is limited)
- Large state and action space
- Two actions per turn for a player (placing the previous tile on the board and selecting a tile from the current tiles)

### Current state of the project

- Kingdomino is implemented
- MPC agent implemented
- Naive DQN agent which takes the state as a vector (no exploiting of regularities) and outputs one of the very numerous actions (where an action is choosing a tile + selecting where to place previous tile)
- Presumably better DQN agent processing the boards along with the previous tile of each player to obtain a vector v, then processing current tiles along with v to finally output an action
- Presumably better action strategy where two DQN agents are used sequentially; one for choosing the tile to pick and one for placing the tile (this reduces the action space for each agent)

### Ideas

Here are some of the ideas that I would like to try:

- Exploit Kingdomino's board symmetries: rotation, reflection
  - Average over group ? (but loss of information probably)
  - Rotation invariant CNNs (+ positional encoding as position matters in Kingdomino) ?
- Change board input to be a set of areas + use some permutation invariant architecture over this set (Transformers, Deep Sets)
- Pre-training of actor and critic (or just a DQN) on MPC-played games
- Training to generalize to several players after pre-training against one:
 - Train against one player, freeze architecture, give last layer (applied to every opponent) as input to small network trained against several players
- JAX implementation of Kingdomino: speed of execution of the environment is a big issue in this project and in RL in general, as it is run on CPU, so a JAX implementation where the environment could be run on GPU could be very valuable
