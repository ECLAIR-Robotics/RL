import gym
import numpy as np
import random
import math
from bandit import BanditTenArmedGaussian

random.seed(21)
np.random.seed(42)

env = BanditTenArmedGaussian()
env.seed(70)

NUM_BANDITS = 10
EPSILON = 0.9

# keeps track of the reward for each bandit lever
q_values = np.ones(NUM_BANDITS)
# number of times this lever gets pulled (this is set to be 1 because UCB can't divide by 0)
n_values = np.ones(NUM_BANDITS)     

# reset the environment
observation = env.reset()

# epsilon greedy (exploitation vs exploration)
def epsilon_greedy():
    # COMPLETE THIS FUNCTION
    return np.random.randint(0, NUM_BANDITS)

# Upper Confidence Bound (UCB)
def upper_confidence_bound(timestep):
    # COMPLETE THIS FUNCTION
    return np.random.randint(0, NUM_BANDITS)

# iterate through the number of timesteps
for timestep in range(0, 500):

    # Pick either epsilon-greedy or UCB
    action = epsilon_greedy()
    #action = upper_confidence_bound(timestep)

    observation, reward, done, info = env.step(action)

    # update the values for q_values and n_values
    # remember the formula for q values is:
    #   q_n+1 = q_n + (1/n)(r_n - q_n)
    # n_values[action] = ?
    # q_values[action] = ?

env.close()
if (np.argmax(q_values) == 6):
    print(f'The lever that yield the most amount of money was lever #{np.argmax(q_values)}!')
    print("Congrats! You're well on your way to becoming a great RL Engineer! :D")
