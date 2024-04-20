###### Introduction

This repository is an implementation of the board game Kingdomino, on which I intend to apply deep reinforcement learning techniques as a way to challenge myself and perhaps find interesting new strategies for the game.
A video explaining the rules of the game can be found here: https://www.youtube.com/watch?v=smbwBPmP4Ms&ab_channel=WatchItPlayed.
Some of the questions I wish to answer is whether it is a better strategy to mess with the plans of other players rather than maximizing one's own score, whether player should focus more often on picking small scores tiles as a way to play first more often, etc.
Previous work on Kingdomino playing agents using Monte-Carlo methods, done by Gedda M. et al: https://arxiv.org/abs/1807.04458.

###### Reasons for Kingdomino

Here is a couple reasons why Kingdomino is an interesting and challenging game:

- Multimodal data: 2D data (boards) as well as tabular data (tiles)
- Multiplayer (although the interaction between players is limited)
- Large state and action space
- Two actions to be done per turn for a player

###### Ideas

Here are some of the ideas that I would like to try:

- Exploit Kingdomino's board symmetries: rotation, reflection
  - Average over group ? (but loss of information probably)
  - Rotation invariant CNNs (+ positional encoding as position matters in Kingdomino) ?
- Change board input to be a set of areas + use some permutation invariant architecture over this set (Transformers, Deep Sets)
- Pre-training of actor and critic (or just a DQN) on MPC-played games
- Training to generalize to several players after pre-training against one:
 - Train against one player, freeze architecture, give last layer (applied to every opponent) as input to small network trained against several players
