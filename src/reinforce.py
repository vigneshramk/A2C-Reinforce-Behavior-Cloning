import sys
import argparse
import numpy as np
import gym
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

# Helper function to convert numpy array to pytorch Variable
def np_to_variable(x, requires_grad=False, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
    if is_cuda:
        v = v.cuda()
    return v

# Helper function to initialize network weights
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

def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v

class Policy(nn.Module):
    def __init__(self, state_size, action_size, grad_clip_range):
        super(Policy, self).__init__()
        self.hidden_size = 16
        self.grad_clip_range = grad_clip_range

        self.classifier = nn.Sequential(
                          nn.Linear(state_size, self.hidden_size),
                          nn.ReLU(),
                          nn.Linear(self.hidden_size, self.hidden_size),
                          nn.ReLU(),
                          nn.Linear(self.hidden_size, self.hidden_size),
                          nn.ReLU(),
                          nn.Linear(self.hidden_size, action_size))

    def forward(self, x):
        x = self.classifier(x)
        x = clip_grad(x, -self.grad_clip_range, self.grad_clip_range)
        return F.softmax(x, dim=1)

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        print('Finished initializing')
          
    def train(self, env, gamma=0.99):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        # Generate data
        states, actions, rewards, log_probs = self.generate_episode(self.model, env)

        # Numpy array creation
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

        G_th = np_to_variable(G)
        #Define the loss and do model.fit here
        hadamard_prod = []
        for log_prob, G_i in zip(log_probs, G_th):
            hadamard_prod.append(-log_prob * G_i)

        # Perform the backpropogation
        self.optimizer.zero_grad()
        loss = torch.cat(hadamard_prod).mean()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
        self.optimizer.step()   

        return G[0], loss, np.sum(rewards)

    def generate_episode(self, model, env, render=False):
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
            action_probs = model(s_th)
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
            states,actions,rewards,log_probs = self.generate_episode(self.model, env)
            reward_epi.append(np.sum(rewards))

        reward_epi = np.array(reward_epi)

        return np.mean(reward_epi), np.std(reward_epi)

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
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
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')

    # Set the seeds
    torch.manual_seed(2018)
    np.random.seed(2018)
    env.seed(2018)

    # Set gamma
    gamma = 0.99

    # Create plot
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.set_title('Per episode Return Plot')

    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.set_title('Test Reward Plot')

    path_name = './fig_reinforce'

    plot1_name = os.path.join(path_name,'reinforce_training_reward.png')
    plot2_name = os.path.join(path_name,'reinforce_test_reward.png')

    # Create plot dir
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    # TODO: Train the model using REINFORCE and plot the learning curve.
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("State_size:{}, Action_size{}".format(state_size, action_size))
    policy = Policy(state_size, action_size,1)
    policy.cuda()
    policy.train()

    reinforce = Reinforce(policy,lr=lr)

    for i in range(num_episodes):
        disc_reward, loss, reward = reinforce.train(env,gamma)
        reward *= 100
        disc_reward *= 100

        print("Total Reward for episode %s is %1.2f" %(i,reward))

        #Test every 500 episodes
        if i % 500 == 0:
            policy.eval()
            mean_r, std_r = reinforce.test(env)
            policy.train()
            print('Episode %s - Mean - %1.2f  Std - %1.2f' %(i,mean_r,std_r))
            ax2.errorbar(i+1, mean_r, yerr=std_r, fmt='o')

            # Save the plots
            str_path2 = 'reinforce_test_reward' + str(i) + '.png'
            plot2_name = os.path.join(path_name,str_path2)
            ax2.figure.savefig(plot2_name)

        # Plot the total reward per episode
        ax1.scatter(i, reward)
        if i%1000 == 0:
            str_path1 = 'reinforce_training_reward' + str(i) + '.png'
            plot1_name = os.path.join(path_name,str_path1)
            ax1.figure.savefig(plot1_name)
            

if __name__ == '__main__':
    main(sys.argv)
