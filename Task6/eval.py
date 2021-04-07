#!/bin/bash
import argparse

import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import torch
import wrappers
import agent

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, help="episode to start tracking number of episodes from", default=20)
parser.add_argument("--game", help="game of the atari environment", default="PongNoFrameskip-v4")
parser.add_argument("--render", help="render the atari emulator", action="store_true")
parser.add_argument("--clip_rewards", help="clip_rewards for atari environment", action="store_false")
parser.add_argument("--frame_stack", type=int, help="number of frames to stack", default=4)
parser.add_argument("--frame_skip", type=int, help="number of frames to skip", default=4)
parser.add_argument("--model_path", help="load the model from directory given", default="agent_saves/DQN_model.pt")

args = parser.parse_args()


episodes = args.episodes
game = args.game
render = args.render
clip_rewards = args.clip_rewards
frame_stack = args.frame_stack
frame_skip = args.frame_skip
model_path = args.model_path

# make the environment, agent, restore weights from path
env = wrappers.make_env(game, clip_rewards=clip_rewards, frame_stack=frame_stack, frame_skip=frame_skip)
agent = agent.Agent()
agent.eval(path=model_path)

# Start env
episode_reward = 0

state = env.reset()
if render:
    env.render()

rewards = []

for episode in range(episodes):
    print('Current episode: ', episode)
    done = False
    while not done:
        action = agent._get_action(state)
        next_state, reward, done, info = env.step(action)

        if render:
            env.render()

        # update variables for next iteration
        state = next_state
        episode_reward += reward

        # We've completed one episode, reset the environment
        if done:
            rewards.append(episode_reward)
            state = env.reset()
            episode_reward = 0

if not os.path.exists('evaluation'):
    os.makedirs('evaluation')
with open('evaluation/evaluation_rewards.csv', 'w+') as csvfile:
    csvwriter = csv.writer(csvfile)
    for reward in rewards:
        csvwriter.writerow([reward])


# graph
fig = plt.figure()
plt.plot(range(episodes), rewards, linestyle='--', marker='o')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.xlim(0, episodes)
plt.ylim(-21, 21)
fig.savefig('evaluation/Reward_per_Episode.png', dpi=fig.dpi)
plt.show()
average = np.mean(np.array(rewards))
print('Number of episodes evaluated: ', episodes)
print('Average Reward: ', average)
