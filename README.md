# imitation.py usage


usage: imitation.py [-h] [--model-config-path MODEL_CONFIG_PATH]
                    [--expert-weights-path EXPERT_WEIGHTS_PATH]
                    [--render | --no-render]

optional arguments:
  -h, --help            show this help message and exit
  --model-config-path MODEL_CONFIG_PATH
                        Path to the model config file.
  --expert-weights-path EXPERT_WEIGHTS_PATH
                        Path to the expert weights file.
  --render              Whether to render the environment.
  --no-render           Whether to render the environment.

eg. python imitation.py --model-config-path ./LunarLander-v2-config.json --expert-weights-path ./LunarLander-v2-weights.h5

# reinforce.py usage

usage: reinforce.py [-h] [--num-episodes NUM_EPISODES] [--lr LR]
                    [--render | --no-render]

optional arguments:
  -h, --help            show this help message and exit
  --num-episodes NUM_EPISODES
                        Number of episodes to train on.
  --lr LR               The learning rate.
  --render              Whether to render the environment.
  --no-render           Whether to render the environment.

e.g. python reinforce.py --num-episodes 50000 --lr 0.0005

# a2c.py usage

usage: a2c.py [-h] [--num-episodes NUM_EPISODES] [--actor-lr ACTOR_LR]
              [--critic-lr CRITIC_LR] [--n N] [--render | --no-render]

optional arguments:
  -h, --help            show this help message and exit
  --num-episodes NUM_EPISODES
                        Number of episodes to train on.
  --actor-lr ACTOR_LR   The actor's learning rate.
  --critic-lr CRITIC_LR
                        The critic's learning rate.
  --n N                 The value of N in N-step A2C.
  --render              Whether to render the environment.
  --no-render           Whether to render the environment.

  e.g. python a2c.py --num-episodes 50000 --actor-lr 1e-3 --critic-lr 1e-3 --n 50