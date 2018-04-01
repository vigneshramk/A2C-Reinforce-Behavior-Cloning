import sys
import argparse
import numpy as np
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

class Policy(nn.Modules):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.hidden_size = 16
        self.classifier = nn.Sequential(
                          nn.Linear(state_size, self.hidden_size),
                          nn.Relu(inplace=True),
                          nn.Linear(state_size, self.hidden_size),
                          nn.Relu(inplace=True),
                          nn.Linear(state_size, self.hidden_size),
                          nn.Relu(inplace=True),
                          nn.Linear(self.hidden_size, action_size))

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x, dim=1)

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=lr,
                                         momentum=0.9)

        # Setting the batch size
        self.batch_size = 32
        self.logger = Logger('./tmp/Reinforce', 'Policy_Gradients')

        print('Finished initializing')

          

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        states,actions,rewards = self.generate_episode(env)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)*1e-2

        sampled_action = actions.copy()

        T = len(rewards)
        G = np.zeros(T)

        for t in range(T)[::-1]:

            K = range(t,T)
            gamma_vec = [pow(gamma,k-t) for k in K]
            # Returns for each step back from the goal
            G[t] = np.sum(rewards[t-T:]*gamma_vec)  

        #Define the loss and do model.fit here
        print("Probs:{}, Actions:{}".format())
        # loss = np.mean(np.dot(G, ))


        return

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        s = env.reset()
        s = np.array(s)
        done = False
        while(done != True):
            
            s = np.reshape(s,[1,8])
            action_softmax = model.predict(s)
            action = np.argmax(action_softmax)
            action_1hot = to_categorical(action, num_classes=4)
            nexts, reward, done, _ = env.step(action)
            nexts = np.array(nexts)
            
            # Append the s,a,r for the current time-step
            states.append(s)
            actions.append(action_1hot)
            rewards.append(reward)

            s = nexts

        return states, actions, rewards

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # print env.observation_space.high
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.


if __name__ == '__main__':
    main(sys.argv)
