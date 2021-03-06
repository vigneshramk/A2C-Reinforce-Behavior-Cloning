import sys
import argparse
import numpy as np
import gym
import os

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def np_to_variable(x, requires_grad=False, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
    if is_cuda:
        v = v.cuda()
    return v

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.hidden_size = 16

        self.classifier = nn.Sequential(
                          nn.Linear(state_size, self.hidden_size),
                          nn.Tanh(),
                          nn.Linear(self.hidden_size, self.hidden_size*2),
                          nn.Tanh(),
                          nn.Linear(self.hidden_size*2, action_size))

        # self.classifier = nn.Sequential(
        #                   nn.Linear(state_size, self.hidden_size),
        #                   nn.ReLU(inplace=True),
        #                   nn.Linear(self.hidden_size, self.hidden_size),
        #                   nn.ReLU(inplace=True),
        #                   nn.Linear(self.hidden_size, self.hidden_size),
        #                   nn.ReLU(inplace=True),
        #                   nn.Linear(self.hidden_size, action_size))

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x, dim=1)

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.


    def __init__(self, model, lr):
        self.model = model

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        print('Finished initializing')
          
    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        states,actions,rewards, log_probs = self.generate_episode(env)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)*1e-2

        T = len(rewards)
        G = np.zeros(T)
        
        for t in range(T)[::-1]:

            K = range(t,T)
            gamma_vec = [pow(gamma,k-t) for k in K]
            # Returns for each step back from the goal
            G[t] = np.sum(rewards[t-T:]*gamma_vec)

        G_normalized = torch.Tensor(G)
        G_normalized = (G_normalized - G_normalized.mean()) / (G_normalized.std() + np.finfo(np.float32).eps)

        #Define the loss and do model.fit here
        # print("Probs:{}, Actions:{}".format())
        # print(type(log_probs))
        hadamard_prod = []
        for log_prob, G_norm in zip(log_probs, G_normalized):
            hadamard_prod.append(-log_prob * G_norm)

        self.optimizer.zero_grad()
        loss = torch.cat(hadamard_prod).sum()
        loss.backward()
        self.optimizer.step()   

        return G[0], loss, np.sum(rewards)

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
        log_probs = []

        s = env.reset()
        s = np.array(s)
        done = False

        num_steps = 0
        while(done != True):
            num_steps += 1
            s = np.reshape(s,[1,len(s)])
            s_th = np_to_variable(s, requires_grad=False)
            action_probs = self.model(s_th)
            action_softmax = Categorical(action_probs)

            #Sample action according to the softmax distribution
            action_sample = action_softmax.sample()
            log_prob = action_softmax.log_prob(action_sample)
            # Use this action to step through the episode
            action = action_sample.data
            action = action[0]
            
            nexts, reward, done, _ = env.step(action)
            nexts = np.array(nexts)
            
            # Append the s,a,r for the current time-step
            states.append(s)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            s = nexts

        #print("Num steps:{}".format(num_steps))

        return states, actions, rewards, log_probs

    def test(self, env, num_episodes = 100,model_file=None):
        
        reward_epi = []
        for i in range(num_episodes):

            states,actions,rewards,log_probs = self.generate_episode(env)
            reward_epi.append(np.sum(rewards))

        reward_epi = np.array(reward_epi)

        return np.mean(reward_epi), np.std(reward_epi)

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
    env = gym.make('CartPole-v0')

    # Set the seeds
    torch.manual_seed(2018)
    np.random.seed(2018)
    env.seed(2018)

    num_episodes = 10000
    gamma = 0.99

    # Create plot
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.set_title('Per episode Cum. Return Plot')

    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.set_title('Test Reward Plot')

    path_name = './fig'
    plot1_name = os.path.join(path_name,'Cartpole_reinforce_training_reward.png')
    plot2_name = os.path.join(path_name,'Cartpole_reinforce_test_reward.png')

    # Create plot dir
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    # TODO: Train the model using REINFORCE and plot the learning curve.
    state_size = env.observation_space.shape[0]
    action_size = 2
    print("State_size:{}, Action_size{}".format(state_size, action_size))
    policy = Policy(state_size, action_size)
    # weights_normal_init(policy, dev=0.01)
    policy.cuda()
    policy.train()

    reinforce = Reinforce(policy,lr=0.001)

    for i in range(num_episodes):
        cum_reward, loss, reward = reinforce.train(env,gamma)
        reward *= 100
        cum_reward *= 100

        print("Rewards for episode %s is %1.2f" %(i,reward))
        print("Loss for episode %s is %1.2f" %(i,loss))

        #Test every 200 episodes
        if i % 100 == 0:
            mean_r, std_r = reinforce.test(env)
            ax2.errorbar(i, mean_r, yerr=std_r, fmt='o')

        # Plot the discounted reward per episode
        ax1.scatter(i, cum_reward)                
        if i%200 == 0:
            ax1.figure.savefig(plot1_name)
            ax2.figure.savefig(plot2_name)

if __name__ == '__main__':
    main(sys.argv)
