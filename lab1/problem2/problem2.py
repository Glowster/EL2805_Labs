# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import pickle


# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

#Parameters
eta = np.array([[1,1], [2, 2], [1, 2], [2, 1], [0, 1], [0, 0]]) # Fourier basis
lam = 0.8 # for eligibility trace
gamma = 1.0 # discount factor
N_episodes = 200 # Number of episodes to run for training
EPS = 0.05 # base exploration rate
ALPHA = 0.003 # base learning rate

lr_lookback = 5 # learning rate reduction lookback
lr_threshold = -160 # threshold for learning rate reduction
# factor to multiply lr with if mean over last look_back episode rewards is less than lr_threshold:
lr_factor = 0.1

mu = 0.5 # SGD momentum/nesterov factor

use_ucb = True # Use UCB exploration. If false, use constant EPS instead
grid_n = 20 # Grid size for pseudo-counts
ucb_scale_param = 1 # Scaling factor in UCB maximization


params = {
    "eta": tuple(tuple(int(x) for x in row) for row in eta),
    "lam": lam,
    "gamma": gamma,
    "N_episodes": N_episodes,
    "EPS": EPS,
    "ALPHA": ALPHA,
    "lr_lookback": lr_lookback,
    "lr_threshold": lr_threshold,
    "lr_factor": lr_factor,
    "mu": mu
}

# Reward
episode_reward_list = []  # Used to save episodes reward



# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x


class Model:

    def __init__(self, eta, lam, gamma, mu, weight_init="zeros"):
        self.eta = eta #(m,2)
        self.m = eta.shape[0]
        self.weights = self.__init_weights(self.m, weight_init) #(m,3)
        self.z = np.zeros((self.m,3))
        self.lam = lam
        self.gamma = gamma
        self.v = np.zeros((self.m,3))
        self.mu = mu

        eta_norm = np.linalg.norm(self.eta, axis=1)
        self.inv_eta_norm = np.diag(1/np.where(eta_norm > 0, eta_norm, 1)) # (m,m) diag
        

    def forward(self, s):
        return self.weights.T @ np.cos(np.pi*self.eta @ s) # (3,)

    def __init_weights(self, m, weight_init):
        if weight_init == "zeros":
            return np.zeros((m,3))
        elif weight_init == "gaussian":
            return  np.random.randn(m,3)
        elif weight_init == "ones":
            return np.ones((m,3))
        elif weight_init == "minus_ones":
            return -np.ones((m,3))
        else:
            raise Exception("Unknown weight init")
    
    def eps_greedy(self, s, eps):
        if np.random.rand() < 1-eps:
            return np.argmax(self.forward(s))
        else:
            return np.random.randint(3)
    
    def ucb(self, s, time, ucb_scale_param, pseudocounts):
        return np.argmax(self.forward(s) + ucb_scale_param * np.sqrt(np.log(time)/pseudocounts))

    def update_trace(self, s, a):
        self.z *= self.gamma*self.lam
        self.z[:, a] += np.cos(np.pi*self.eta @ s)
        self.z = np.clip(self.z, -5, 5)

    def Phi(self, s):
        return np.cos(np.pi*self.eta @ s)

    def Q(self,s,a):
        return (self.weights.T @ self.Phi(s))[a]

    def update_weights(self, s, a, r, s_next, a_next, alpha):
        delta = (r+self.gamma*self.Q(s_next, a_next)-self.Q(s,a))
        self.weights +=  alpha*delta*(self.inv_eta_norm @ self.z)

    def update_weights_momentum(self, s, a, r, s_next, a_next, alpha):
        delta = (r+self.gamma*self.Q(s_next, a_next)-self.Q(s,a))
        self.v = self.mu * self.v + alpha*delta*(self.inv_eta_norm @ self.z)
        self.weights += self.v
    
    def update_weights_nesterov(self, s, a, r, s_next, a_next, alpha):
        delta = (r+self.gamma*self.Q(s_next, a_next)-self.Q(s,a))
        self.v = self.mu * self.v + alpha*delta*(self.inv_eta_norm @ self.z)
        self.weights += self.mu * self.v + alpha*delta*(self.inv_eta_norm @ self.z)
    

    def reset_trace(self):
        self.z = np.zeros((self.m,3))
        self.v = np.zeros((self.m,3))

    def save(self, file_name):
        d = {}
        d["N"] = self.eta
        d["W"] = self.weights.T

        with open(file_name, 'wb') as file:
            pickle.dump(d, file)


model = Model(eta, lam, gamma, mu) 
episode_reward_list = []
last_epsiode_pos = []
last_epsidoe_vel = []

if use_ucb:
    grid_counts = np.ones((3, grid_n+1, grid_n+1), dtype=np.int64)

# Training process
t = 1
for i in range(N_episodes):
    # Reset enviroment data
    done = False
    truncated = False
    state = scale_state_variables(env.reset()[0])
    total_episode_reward = 0.

    eps = EPS

    alpha = ALPHA
    if i > lr_lookback and lr_threshold < np.mean(episode_reward_list[-lr_lookback:]):
        alpha *= lr_factor

    model.reset_trace()

    if use_ucb:
        action = model.ucb(state, t, ucb_scale_param, grid_counts[:,*map(int, np.floor(state*grid_n))])
    else:
        action =  model.eps_greedy(state, eps)

    while not (done or truncated):
        t += 1
        if i == N_episodes-1:
            last_epsiode_pos.append(state[0])
            last_epsidoe_vel.append(state[1])

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = scale_state_variables(next_state)

        if use_ucb:
            next_action = model.ucb(state, t, ucb_scale_param, grid_counts[:,*map(int, np.floor(state*grid_n))])
        else:
            next_action = model.eps_greedy(next_state, eps)

        model.update_trace(state, action)
        model.update_weights_momentum(state, action, reward, next_state, next_action, alpha)
        
        # Update episode reward
        total_episode_reward += reward

        if use_ucb:
            grid_counts[action, *map(int, np.floor(state*grid_n))] += 1
        
        # Update state for next iteration
        state = next_state
        action = next_action

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()
    

model.save("weights.pkl")

# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.axhline(-135, color='r', linestyle='--', label='y = -135')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

