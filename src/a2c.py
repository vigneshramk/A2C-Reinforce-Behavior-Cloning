import sys
import argparse
import numpy as np
import gym
import os
from copy import copy, deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce

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

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class Value(nn.Module):
    def __init__(self, state_size, action_size):
        super(Value, self).__init__()
        self.hidden_size = 16
        self.classifier = nn.Sequential(
                          nn.Linear(state_size, self.hidden_size),
                          nn.ReLU(),
                          nn.Linear(self.hidden_size, self.hidden_size*4),
                          nn.ReLU(),
                          nn.Linear(self.hidden_size*4, self.hidden_size*16),
                          nn.ReLU(),
                          nn.Linear(self.hidden_size*16, self.hidden_size*4),
                          nn.ReLU(),
                          nn.Linear(self.hidden_size*4, 1))

    def forward(self, x):
        x = self.classifier(x)
        return x

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, actor_model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.actor_model = deepcopy(actor_model)
        self.critic_model = deepcopy(critic_model)
        self.n = n

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

        self.optimizer_actor = optim.Adam(self.actor_model.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic_model.parameters(), lr=critic_lr)
        self.loss_critic = nn.MSELoss()


        print('Finished initializing') 

    def train(self, env, gamma=0.99):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        states,actions,rewards, log_probs = self.generate_episode(self.actor_model, env)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)*1e-2
        
        T = len(rewards)
        R = np.zeros(T)
        N_vec = range(self.n)
        gamma_vec = [pow(gamma,n) for n in N_vec]
        V_vec = np.zeros(T)
        V_vec_th_critic = []

        for t in range(T):
            s_curr = np_to_variable(states[t], requires_grad=False)
            V_vec_th_critic.append(self.critic_model(s_curr))
            V_vec[t] = V_vec_th_critic[t].data.cpu().numpy()[0]

        V_vec_th_critic = torch.cat(V_vec_th_critic)
        
        # Compute the N-step rewards
        for t in range(T)[::-1]:
            if (t + self.n) >= T:
                V_end = 0
            else:
                V_end = V_vec[t+self.n]
            r_n = np.zeros(self.n)
            for k in range(self.n):
                if(t+k) < T:
                    r_n[k] = rewards[t+k]
            R[t] = np.power(gamma,self.n)*V_end + np.dot(gamma_vec,r_n)

        #Cum rewards vector for each episode
        R = np_to_variable(R, requires_grad=False)
        V_vec_th_actor = np_to_variable(V_vec, requires_grad=False)
        cum_R = R - V_vec_th_actor

        #Loss definition for the actor and critic networks
        #Actor

        # Actor update
        hadamard_prod = []
        for l_prob, c_R in zip(log_probs, cum_R):
            hadamard_prod.append(-l_prob * c_R)

        self.optimizer_actor.zero_grad()
        loss_actor = torch.mean(torch.cat(hadamard_prod))
        loss_actor.backward()
        nn.utils.clip_grad_norm(self.actor_model.parameters(), 2)
        self.optimizer_actor.step()

        # Critic update
        self.optimizer_critic.zero_grad()
        loss_th_critic = self.loss_critic(V_vec_th_critic,R)
        loss_th_critic.backward()
        nn.utils.clip_grad_norm(self.critic_model.parameters(), 2)
        self.optimizer_critic.step()

        return np.sum(rewards), loss_actor, loss_th_critic.data[0]

    def test(self, env, num_episodes = 100,model_file=None):
        
        reward_epi = []
        for i in range(num_episodes):

            states,actions,rewards,log_probs = self.generate_episode(self.actor_model, env)
            reward_epi.append(np.sum(rewards))

        reward_epi = np.array(reward_epi)

        return np.mean(reward_epi), np.std(reward_epi)

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=50, help="The value of N in N-step A2C.")

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
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Set the seeds
    torch.manual_seed(2018)
    np.random.seed(2018)
    env.seed(2018)

    num_episodes = 50000
    gamma = 0.99
    print(env.observation_space.shape)
    print(env.action_space)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("State_size:{}, Action_size{}".format(state_size, action_size))

    #Define actor and crtiic learning rates here
    actor_lr = 5e-4
    critic_lr = 5e-4

    # # Create plot
    # fig1 = plt.figure()
    # ax1 = fig1.gca()
    # ax1.set_title('Per episode Cum. Return Plot')
    path_name = './fig_a2c_pr3_ar3_n100'
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    # plot1_name = os.path.join(path_name,'reinforce_discounted_reward.png')
    # plot1_name = os.path.join(path_name,'a2c_reward.png')

    # Create plot
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.set_title('Per episode Cum. Return Plot')

    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.set_title('Test Reward Plot')

    # TODO: Train the model using A2C and plot the learning curves.

    #Create instances of actor and critic
    critic = Value(state_size,action_size)
    critic.cuda()
    critic.train()
    actor = Policy(state_size,action_size)
    actor.cuda()
    actor.train()

    a2cAgent = A2C(actor, actor_lr,critic, critic_lr, n)

    for i in range(num_episodes):
        cum_reward, loss_actor, loss_critic = a2cAgent.train(env,gamma)
        cum_reward *= 100
        # plt.pause(0.005)
        if i%100 == 0:
            print("Rewards for episode %s is %1.2f" %(i,cum_reward))
            # print("Loss Actor for episode %s is %1.2f" %(i,loss_actor))
            # print("Loss Critic for episode %s is %1.2f" %(i,loss_critic))

        #Test every 300 episodes
        if i % 1000 == 0:
            mean_r, std_r = a2cAgent.test(env)

            print('Episode %s - Mean - %1.2f  Std - %1.2f' %(i,mean_r,std_r))
            ax2.errorbar(i+1, mean_r, yerr=std_r, fmt='o')

        # Plot the discounted reward per episode
        ax1.scatter(i, cum_reward)
        if i%100 == 0:
            str_path1 = 'a2c_training_reward' + str(i) + '.png'
            str_path2 = 'a2c_test_reward' + str(i) + '.png'
            plot1_name = os.path.join(path_name,str_path1)
            plot2_name = os.path.join(path_name,str_path2)
            ax1.figure.savefig(plot1_name)
            ax2.figure.savefig(plot2_name)

if __name__ == '__main__':
    main(sys.argv)
