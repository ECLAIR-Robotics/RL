import gym
import gym_gridworlds
import numpy as np
import random
import math

random.seed(21)
np.random.seed(42)

env = gym.make('Cliff-v0')
env.seed(70)

MAP_SIZE      = [4, 12]
STATES        = MAP_SIZE[0] * MAP_SIZE[1]       # 70 possible states
ACTIONS       = 4                               # Can either go up, right, down, or left from each state (0, 1, 2, 3)
STEP_SIZE     = 0.1
DISCOUNT_RATE = 0.9
EPSILON       = 0.8
action_value = np.zeros((STATES, ACTIONS))

# Args:
# - state: tuple of (x,y) coordinate
# Returns:
# - Flattened version so that action_value can be 2 dimensional
def convert_state_tuple(state):
    (x,y) = state
    return (x * MAP_SIZE[1]) + y

def epsilon_greedy(actions):
    r = random.random()

    if r < EPSILON:
        # greedy option
        action = np.argmax(actions)
    else:
        # exploratory option
        action = np.random.randint(0, ACTIONS)
    return action

# episode is how many times something runs until our sequences reaches termination
for episode in range(0, 500):

    # reset the environment (SA___ part of "SARSA")
    observation = env.reset()
    state = convert_state_tuple(observation)
    action = epsilon_greedy(action_value[state])

    done = False

    while not done:            
        # __RSA part of "SARSA"
        observation, reward, done, _ = env.step(action)

        # TODO: Implement SARSA here!
        ################# YOUR CODE ################# 



        ################# YOUR CODE ################# 

val_map  = np.reshape(np.argmax(action_value, 1), (MAP_SIZE[0], MAP_SIZE[1]))
game_map = np.zeros((MAP_SIZE[0], MAP_SIZE[1]))

print("Optimal action for each state (0-up, 1-right, 2-down, 3-left):")
print(np.round(val_map))

# create map to make sure everything is good
i = 3
j = 0
game_map[i][j] = 1

done = False
count = 2
observation = env.reset()
state = convert_state_tuple(observation)

while not done:
    # up, right, down, left
    val = action_value[state]

    # find the optimal policy at the given position
    optimal = np.argmax(val)
    observation, reward, done, _ = env.step(optimal)
    (i, j) = observation
    state = convert_state_tuple(observation)

    game_map[i][j] = count
    count += 1

    if i == MAP_SIZE[0] - 1 and j == MAP_SIZE[1] - 1:
        done = True

print("Optimal path generated:")
print(game_map)

# Should have the game map of the following. This is the safest policy but not the quickest!
# 
# [[ 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15.]
#  [ 3. 0. 0. 0. 0. 0.  0.  0.  0.  0.  0. 16.]
#  [ 2. 0. 0. 0. 0. 0.  0.  0.  0.  0.  0. 17.]
#  [ 1. 0. 0. 0. 0. 0.  0.  0.  0.  0.  0. 18.]]

# What's the difference between a state value function and an action value function
# TODO: PUT YOUR ANSWER BELOW
#
# A: ?