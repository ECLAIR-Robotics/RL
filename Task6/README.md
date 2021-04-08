# DQN for playing Atari

## Dependencies:
* [gym](https://github.com/openai/gym) (atari)

* [PyTorch](https://github.com/pytorch/pytorch)

## Creating the Python Environment:

### Manual Anaconda:
```bash
conda create -n atariEnv -y python=3.8
conda activate atariEnv
pip install gym[atari]
pip install matplotlib
```

#### PyTorch
For CPU users:
```bash
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
```

For CUDA users:
```bash
conda install -y pytorch torchvision torchaudio cudatoolkit=<CUDA_VERSION> -c pytorch
```

### Automatic Anaconda:
Note this installs `cudatoolkit=10.1` and maybe some extra libraries that are not needed to run the code
```
conda env create -f atariEnv.yml
```

## Overview
```
.
└── Atari-DQN/
    ├── agent.py ——————————— Class representing our agent, which contains the DQN functionality
    ├── atariEnv.yml ——————— .yml file for creating the atariEnv
    ├── eval.py ———————————— Used to evaluate an agent, given model weights to load
    ├── memory.py —————————— Class representing our Experience Replay memory buffer component used by the agent to store transition sequences
    ├── model.py ——————————— Class representing the neural network used by our agent to approximate the Q function written in PyTorch
    ├── README.py —————————— This document
    ├── run.py ————————————— Used to train/resume training an agent
    └── wrappers.py ———————— Various wrappers for the gym atari environment used to implemenent specific environment conditions (grayscale, frame stacking, frame skipping, etc)
```

You will only need to edit `agents.py` for this task. Specifically, the function
`compute_td_loss` which calculates the loss. If you want to change up other
things, feel free to send a message in the Discord chat!

To run the DQN, run the command `python run.py.` You can change the some hyperparameters by altering some flags if you want. You can read `run.py` to see all the flags. 


## References
* [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), [Original code](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)

* [OpenAI Baselines](https://github.com/openai/baselines)

* [RL-Adventure](https://github.com/higgsfield/RL-Adventure)
