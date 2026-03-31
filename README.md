# 6x6 Checkers with Actor-Critic (Self-Play)

## Task 1: Custom Environment

The board is represented as a 6×6 grid with values: 0 (empty), 1/-1 (pieces), and 2/-2 (kings). The action space is a tuple (from_row, from_col, to_row, to_col), with only legal moves allowed. Captures are mandatory, and pieces are promoted to kings upon reaching the opposite side.

Rewards: +1 for win, -1 for loss, +0.25 for capture, and 0.5 each for a draw (only kings remain). The game ends when a player has no pieces, no legal moves, or a draw occurs.

## Task 2: Actor-Critic & Self-Play

Actor-Critic uses two components: an actor that learns the policy and the critic that estimates the value function and is updated using the TD error.
The critic computes the TD error δ = r + γV(s′) − V(s) and updates the value function.  
The actor updates the policy using log π(a|s) weighted by δ.  
The negative sign is used because optimization minimizes loss, while policy gradients maximize reward.