import random
import numpy as np
import torch

import mycheckersenv
from myagent import ACAgent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_self_play(num_episodes=2000, seed=42):
    set_seed(seed)

    env = mycheckersenv.env(render_mode=None)
    agent = ACAgent(entropy_coef=0.01)

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        env.reset(seed=seed + episode)
        total_reward = 0.0

        for current_agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            total_reward += reward

            if done:
                env.step(None)
                continue

            legal_moves = info.get("legal_moves", [])
            action, log_pi, v, entropy = agent.select_action(
                obs, legal_moves, current_agent
            )

            env.step(action)                                #Agent selected action using policy pi
            step_reward = env.rewards[current_agent]        #Immediate Reward
            next_obs = env.observe(env.agent_selection)     #Next State(S')
            next_done = env.terminations[env.agent_selection] or env.truncations[env.agent_selection]   #Checking if game is completed

            #Next Value function is found out(V')
            if next_done:
                v_next = torch.tensor([0.0], dtype=torch.float32, device=agent.device)
            else:
                v_next = agent.get_value(next_obs, env.agent_selection)

            #The extracted values are sent to the agent for computing the loss
            agent.update(
                log_pi=log_pi,
                v=v,
                reward=step_reward,
                v_next=v_next,
                done=next_done,
                entropy=entropy,
            )

        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100.0
            print(f"Episode {episode}, avg reward (last 100): {avg_reward:.3f}")

    agent.save("ac_checkers.pt")
    env.close()
    return episode_rewards


def demo(seed=999):
    env = mycheckersenv.env(render_mode="human")
    agent = ACAgent()
    agent.load("ac_checkers.pt")

    env.reset(seed=seed)
    cumulative_reward = 0.0
    winner_printed = False

    for current_agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        cumulative_reward += reward

        if termination or truncation:
            action = None
        else:
            legal_moves = info.get("legal_moves", [])
            action, _, _, _ = agent.select_action(obs, legal_moves, current_agent)

        env.step(action)

        if not winner_printed:
            winner = env.infos["player_1"].get("winner")
            if winner is not None:
                print(f"Winner: {winner}")
                winner_printed = True

    print(f"Final cumulative reward: {cumulative_reward:.3f}")
    env.close()


if __name__ == "__main__":
    train_self_play(num_episodes=2000, seed=43)
    demo(seed=123)