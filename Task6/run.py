#!/bin/bash


import argparse
import agent
import wrappers
import os
import csv

# update params to resume from command line
parser = argparse.ArgumentParser()
parser.add_argument("--frame_start", type=int, help="number of frames to skip", default=1)
parser.add_argument("--episode", type=int,help="episode to start tracking number of episodes from", default=0)
parser.add_argument("--game", help="game of the atari environment", default="PongNoFrameskip-v4")
parser.add_argument("--render", help="render the atari emulator", action="store_true")
parser.add_argument("--clip_rewards", help="clip_rewards for atari environment", action="store_false")
parser.add_argument("--frame_stack", type=int,help="number of frames to stack", default=4)
parser.add_argument("--frame_skip", type=int,help="number of frames to skip", default=4)
parser.add_argument("--num_frames", type=int,help="number of frames to train on", default=2000000)
parser.add_argument("--batch_size", type=int,help="size of transition batches to appy Q-learning updates on", default=32)
parser.add_argument("--replay_initial", type=int,help="How many transition states to store in memory before start applying updates", default=1000)
parser.add_argument("--memory_capacity", type=int,help="total number of transitions stores in replay memory at one time", default=100000)
parser.add_argument("--gamma", type=float,help="discount factor applied to calculate sum of future rewards", default=0.99)
parser.add_argument("--update_freq", type=int,help="how often we should update our target model", default=1000)
parser.add_argument("--epsilon_start", type=float,help="starting value of epsilon", default=1.0)
parser.add_argument("--epsilon_final", type=float,help="ending value of epsilon", default=0.01)
parser.add_argument("--epsilon_decay", type=int,help="factor that determines the annealing rate, determines which frame number epsilon will reach epsilon_final at", default=30000)
parser.add_argument("--lr", type=float,help="learning rate for our optimizer function", default=0.0001)
# by default does not load model path at all, if just use command `--load_model` it will load from default directory of "agent_save", specify directory by `--load_model directory_path`
parser.add_argument("--load_model", help="load the model from directory given", nargs='?', const="agent_saves", default=False)
parser.add_argument("--checkpoint_save", type=int,help="how often to save the weights in agent_saves", nargs='?', const="agent_saves", default=10000)
parser.add_argument("--log", help="whether to log losses and rewards in csv file", action="store_true")


args = parser.parse_args()
frame_start = args.frame_start
episode = args.episode

'''
Environment settings:
render: Render the atari screen if True otherwise just do the computations
clip_rewards: Original DQN and subsequent variations clipped overall env reward to {-1, 0, 1} rather than say the score 
              of the pong game which could be between [-21, 21]
frame_stack: Number of frames to bundle together to send through network, this is important for games like Pong, 
             where a single frame doesn't show you what direction the ball is moving, this turns the what would normally 
             be a POMDP to a MDP, read Peter stones paper for more info: https://arxiv.org/pdf/1507.06527.pdf
frame_skip: Number of frames "skipped" by the env, (action is repeated for the 4 frames), used to speed up training

render: Render the atari environment live in an emulator

Note: I wouldn't mess with the grayscaling/RGB setting right now, unless you know how to properly change the convnet/dimensions
      to incorporate this
'''
game = args.game
render = args.render
clip_rewards = args.clip_rewards
frame_stack = args.frame_stack
frame_skip = args.frame_skip

'''
Hyper Parameters:
num_frames: Total number of frames we are going to train on
batch_size: Number of transition sequences (S, A, R, S') we are going to compute our loss with every iteration after replay_initial
replay_initial: Determines how many (S, A, R, S') transitions we should initially load in the memory before starting to calculate the loss function, must be >= batch_size
memory_capacity: Determines the full capacity of our experience replay, how many total transition sequences to store at one time, storing new transitions will overwrite old transitions when capacity is reached
gamma: Discount factor applied to calculate sum of future rewards
update_freq: Determines how often we should update our target model i.e. every X number of frames we refresh it's weights with our current live model, used to stabilize the network and helps it not get stuck in local minima
epsilon_start: What number to start epsilon at
epsilon_final: What number should epsilon end at
epsilon_decay: Factor that determines the annealing rate, determines which frame number epsilon will reach epsilon_final at
epsilon = lambda frame_num: max(epsilon_start - (epsilon_start - epsilon_final) * (frame_num / epsilon_decay), epsilon_final)
lr: Learning rate for our optimizer function
'''
num_frames = args.num_frames
batch_size = args.batch_size
replay_initial = args.replay_initial
memory_capacity = args.memory_capacity
gamma = args.gamma
update_freq = args.update_freq
epsilon_start = args.epsilon_start
epsilon_final = args.epsilon_final
epsilon_decay = args.epsilon_decay
lr = args.lr
epsilon = lambda frame_num: max(epsilon_start - (epsilon_start - epsilon_final) * (frame_num / epsilon_decay), epsilon_final)

load_model = args.load_model
checkpoint_save = args.checkpoint_save
log = args.log

# make the environment
env = wrappers.make_env(game, clip_rewards=clip_rewards, frame_stack=frame_stack, frame_skip=frame_skip)

# Model settings
output_dim = env.action_space.n
input_dims = (frame_stack, 84, 84)

# make the agent
agent = agent.Agent(memory_capacity, gamma, input_dims, output_dim, lr)

if load_model:
    print(load_model)
    agent.load_variables(direc=load_model, copy_model_to_target=True, load_mem=True)

# For tracking results
episode_reward = 0
loss = 0

if log:
    losses = []
    rewards = []
    open('agent_saves/losses.csv', 'w')
    open('agent_saves/rewards.csv', 'w')

# Start env
state = env.reset()
reward = 0
if render:
    env.render()

# Training Loop
for frame in range(frame_start, num_frames + 1):
    action = agent.get_action(state, epsilon(frame), env.action_space.n)
    next_state, reward, done, info = env.step(action)

    # Push this transition sequence to the agents memory buffer
    agent.memory.add(state, action, reward, next_state, done)

    if render:
        env.render()

    # update variables for next iteration
    state = next_state
    episode_reward += reward

    # We've completed one episode, reset the environment
    if done:
        print(episode)
        if log:
            rewards.append(episode_reward)
            if len(rewards) % checkpoint_save == 0:
                with open('agent_saves/rewards.csv', 'a+') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for val in rewards:
                        csvwriter.writerow([val])
                rewards = []
        episode += 1
        state = env.reset()
        episode_reward = 0

    # Only calculate loss when we have replay_initial transition sequences stored
    if len(agent.memory.memory) > replay_initial:
        # compute loss function, and back prop
        loss = agent.compute_td_loss(batch_size)
        if log:
            losses.append(loss.item())
            # write every 100K things to a file
            if len(losses) % checkpoint_save == 0:
                with open('agent_saves/losses.csv', 'a+') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for val in losses:
                        csvwriter.writerow([val])
                losses = []


    # Update the weight of our target network with the weights of our model
    if frame % update_freq == 0:
        agent.update_target()

    # Save the weights of the model, and optimizer every checkpoint_save frames, can be helpful to reload the state
    # training if something happens midway through
    if frame % checkpoint_save == 0:
        agent.save_variables()
        print(episode)
        with open('agent_saves/chkpt.txt', 'w') as file:
            file.seek(0)
            file.write('Variables saved. Default save PATH for model: agent_saves/DQN_model.pt, Default save PATH for optimizer: agent_saves/Adam_optimizer.pt\n')
            file.write('-------------------------------------------------------------------------------------------------------\n')
            file.write('Episode: ' + str(episode) + '\n')
            file.write('Current frame number: ' + str(frame)+ '\n')
            file.write('Current epsilon at: ' + str(epsilon(frame))+ '\n')
            file.write('Last loss value calculated to be:' + str(loss)+ '\n')
            file.truncate()

# Save final model weights
agent.save_model(direc='agent_saves', name='DQN.pt')
