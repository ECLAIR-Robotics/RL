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

# keeps track of the reward for each bandit lever
state_value = np.zeros(STATES)

# Args:
# - state: tuple of (x,y) coordinate
# Returns:
# - Flattened version so that state_value can be 2 dimensional
def convert_state_tuple(state):
    (x,y) = state
    return (x * MAP_SIZE[1]) + y

def random_action():
    return np.random.randint(0, ACTIONS)

# reset the environment
observation = env.reset()
state = convert_state_tuple(observation)

# episode is how many times something runs until our sequences reaches termination
for timestep in range(0, 80000):

    # The official algorithm says "A <-- action given by policy for S"
    # THIS IS STUPIDDDDDDDDDDD i hate everything. TD only works on KNOWN
    # policies (aka like 40% to go up, 20% to go down, etc at a given state).
    # For this policy, we're taking random actions so 25% anytime!!!! i dum dum
    #
    # For SARSA, we will compute the policy, but right now it's super
    # simple!
    action = random_action()

    observation, reward, done, _ = env.step(action)

    # TODO: Implement Temporal Difference (TD(0)) here!
    ################# YOUR CODE ################# 



    ################# YOUR CODE ################# 

val_map  = np.reshape(state_value, (MAP_SIZE[0], MAP_SIZE[1]))
game_map = np.zeros((MAP_SIZE[0], MAP_SIZE[1]))
print("Value matrix for each state:")
print(np.round(val_map))

# Should be the following (if it's slightly different, that's okay!):
# [[ -52.  -64.  -64.  -59.  -61.  -70.  -62.  -66.  -53.  -44.  -27.  -21.]
#  [ -72. -102.  -86.  -86.  -93.  -88.  -78.  -92.  -89.  -59.  -42.  -29.]
#  [-104. -132. -131. -142. -150. -140. -140. -149. -122. -118. -133.  -60.]
#  [-147.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  -88.]]

# create map to make sure everything is good
i = 3
j = 0
game_map[i][j] = 1

done = False
count = 2
observation = env.reset()

while not done:
    # up, right, down, left
    possiblities = [[i-1,j],[i,j+1],[i+1,j],[i,j-1]]
    val = np.ones(ACTIONS) * -100000

    for l in range(0, len(possiblities)):
        (x, y) = possiblities[l]
        if x != -1 and y != -1 and x != MAP_SIZE[0] and y != MAP_SIZE[1] and val_map[x][y] != 0 and game_map[x][y] == 0:
            val[l] = val_map[x][y]

    # find the optimal policy at the given position
    optimal = np.argmax(val)
    observation, reward, done, _ = env.step(optimal)
    (i, j) = observation

    game_map[i][j] = count
    count += 1

    if i == MAP_SIZE[0] - 1 and j == MAP_SIZE[1] - 1:
        done = True

print("Optimal path generated:")
print(game_map)

# Should be the following:
# [[ 4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15.]
#  [ 3. 26. 25. 24. 23. 22. 21. 20. 19. 18. 17. 16.]
#  [ 2. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36. 37.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 38.]]
#
# Why do you think it's not the following?
# [[ 0. 0. 0. 0. 0. 0. 0. 0.  0.  0.  0.  0.]
#  [ 0. 0. 0. 0. 0. 0. 0. 0.  0.  0.  0.  0.]
#  [ 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13.]
#  [ 1. 0. 0. 0. 0. 0. 0. 0.  0.  0.  0. 14.]]
#
# (Hint: It has something to do with how we selected our action :D)
# TODO: PUT YOUR ANSWER BELOW
#
# A: ?