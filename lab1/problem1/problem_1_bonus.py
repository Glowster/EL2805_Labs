# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

# Implemented methods
methods = ['Q_learning', 'sarsa']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'


###############################################


class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values 
    STEP_REWARD = 0
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = 0
    MINOTAUR_REWARD = 0

    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0
        m=0
        for m in range(2):
            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[1]):
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            if self.maze[i,j] != 1:
                                states[s] = ((i, j), (k,l), m)
                                map[((i,j), (k,l), m)] = s
                                s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        s += 1

        states[s] = 'Terminal'
        map['Terminal'] = s
        
        return states, map


    def __move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """
        
        if self.states[state] == 'Eaten' or self.states[state] == 'Win' or self.states[state] == 'Terminal': # In these states, the game is over
            return ['Terminal']
        
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
            
            # Is the player getting out of the limits of the maze or hitting a wall?
            impossible_action_player = not((0 <= row_player < self.maze.shape[0]) and (0 <= col_player < self.maze.shape[1]) and self.maze[row_player,col_player] != 1)
            
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]] # Possible moves for the Minotaur
            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                             (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
          

            # Based on the impossiblity check return the next possible states.
            if impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if self.states[state][0][0] == rows_minotaur[i] and self.states[state][0][1] == cols_minotaur[i]:
                        states.append('Eaten')
                    
                    elif self.maze[self.states[state][0][0], self.states[state][0][1]] == 2 and self.states[state][2] == 1:
                        states.append('Win')
                
                    else:     # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))

                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if row_player == rows_minotaur[i] and col_player == cols_minotaur[i]:
                        states.append('Eaten')
                    
                    elif self.maze[row_player,col_player] == 2 and self.states[state][2] == 1:
                        states.append('Win')
                    
                    elif self.maze[row_player, col_player] == 3:

                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), 1))
                    else:
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
              
                return states
        
    
    def next_state(self, state, action):
        moves = self.__move(state, action)
        
        if "Terminal" in moves:
            return "Terminal"

        if np.random.rand() <= 0.35:

            if "Eaten" in moves:
                return "Eaten"

            if "Win" in moves:
                return "Win"

            dists = []
            for move in moves:

                dists.append(np.linalg.norm(np.array(move[0])-np.array(move[1]), ord=1))
            
            dists = np.array(dists)

            closer_inds = np.argwhere(dists == np.min(dists))

            
            moves_closer  = [moves[i[0]] for i in closer_inds]

            return random.choice(moves_closer)
   
        else:
            return random.choice(moves)
        



    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)
  
        for s in range(self.n_states):
            for a in range(self.n_actions):
                possible_moves = self.__move(s,a)
                for s_prime in possible_moves:
                    transition_probabilities[self.map[s_prime],s,a] += 1/len(possible_moves)
    
        return transition_probabilities



    def __rewards(self):
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD

                elif self.states[s] == 'Terminal':
                    continue

                else:                
                    next_states = self.__move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    
                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    
                    else: # Regular move
                        rewards[s, a] = self.STEP_REWARD

        return rewards

    #def step(self, action):
    #    next_states = self.__move(self.states(s, a) 


    def simulate(self, start, policy, method):
        
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        
        if method == 'DynProg':
            horizon = policy.shape[1] # Deduce the horizon from the policy shape
            t = 0 # Initialize current time
            s = self.map[start] # Initialize current state 
            path.append(start) # Add the starting position in the maze to the path
            
            while t < horizon - 1:
                a = policy[s, t] # Move to next state given the policy and the current state
                next_states = self.__move(s, a) 
                next_s = random.choice(next_states)
                path.append(next_s) # Add the next state to the path
                t +=1 # Update time and state for next iteration
                s = self.map[next_s]
                
        if method == 'ValIter': 
            t = 1 # Initialize current state, next state and time
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
            next_s = random.choice(next_states)
            path.append(next_s) # Add the next state to the path
            
            horizon = np.random.geometric(1/30) # Question e
            # Loop while state is not the goal state
            while s != next_s and t <= horizon: # TODO change to is s_win or s_eaten /Viktor
                s = self.map[next_s] # Update state
                next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
                next_s = random.choice(next_states)
                path.append(next_s) # Add the next state to the path
                t += 1 # Update time for next iteration
    
        if method in ['Q_learning', 'sarsa']:
            t = 1 # Initialize current state, next state and time
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            next_s = self.next_state(s, policy[s]) # Move to next state given the policy and the current state
            path.append(next_s) # Add the next state to the path
            
            horizon = np.random.geometric(1/50) # Question e
            
            # Loop while state is not the goal state
            while s != next_s and t < horizon: # TODO change to is s_win or s_eaten /Viktor
                s = self.map[next_s] # Update state
                next_s = self.next_state(s, policy[s]) # Move to next state given the policy and the current state
                path.append(next_s) # Add the next state to the path
                t += 1 # Update time for next iteration

        
        
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI



    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)



def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    V = np.zeros((env.n_states,horizon))
    policy = np.zeros((env.n_states,horizon))

    V[:, horizon-1] = np.max(env.rewards, axis=1)
    policy[:,horizon-1] = np.argmax(env.rewards, axis=1)

    for t in range(horizon-2,-1,-1):
        # max(rewards + transition[s_prime,s,a] * V[s_prime,t+1], axis=a)
        Q = env.rewards + (env.transition_probabilities.T @ V[:,t+1]).T
        V[:,t] = np.max(Q, axis=1)
        policy[:,t] = np.argmax(Q, axis=1)

    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracyhorizon of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S
    """

    V = np.zeros(env.n_states)
    err = epsilon*(1-gamma)/gamma + 1

    while err > epsilon*(1-gamma)/gamma:
        new_V = np.max(env.rewards + gamma*(env.transition_probabilities.T @ V).T, axis=1)
        err = np.linalg.norm(new_V - V)
        V = new_V
    
    policy = np.argmax(env.rewards + gamma*(env.transition_probabilities.T @ V).T, axis=1)

    return V, policy


#Choose action based on epsilon-greedy policy

def choose_action(env, state, q_value, eps):
    if np.random.binomial(1, eps) == 1:
        return np.random.choice(env.n_actions)  # Random action
    else:
        values_ = q_value[state, :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


# Q-Learning update rule
"""
def q_learning(env, q_value, step_size, gamma):
    state = env.reset()[0]  # Initial state
    done, truncated = False, False
    while not (done or truncated):
        action = choose_action(state, q_value)
        next_state, reward, done, truncated, _ = env.step(action)
        q_value[state, action] += step_size * (reward + gamma * np.max(q_value[next_state, :]) - q_value[state, action])
        state = next_state
    return q_value
"""

def q_learning(env, start_state, alpha, q_value, gamma, eps, visits):
    state = env.map[start_state]  # Initial state
    while env.states[state] != "Terminal":
        a = choose_action(env, state, q_value, eps)
        next_s = env.map[env.next_state(state, a)] 
        reward = env.rewards[state, a]
        q_value[state, a] += 1/visits[state, a]**alpha * (reward + gamma * np.max(q_value[next_s, :]) - q_value[state, a])
        visits[state, a] += 1

        state = next_s

    return q_value, visits


def sarsa(env, start_state, alpha, q_value, gamma, eps, visits):
    state = env.map[start_state]  # Initial state
    a = choose_action(env, state, q_value, eps)
    while env.states[state] != "Terminal":
        next_s = env.map[env.next_state(state, a)] 
        reward = env.rewards[state, a]
        next_a = choose_action(env, next_s, q_value, eps)
        q_value[state, a] += 1/visits[state, a]**alpha * (reward + gamma * q_value[next_s, next_a] - q_value[state, a])
        visits[state, a] += 1

        state, a = next_s, next_a

    return q_value, visits

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE, 3: YELLOW}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(0, len(path)):
        if path[i-1] not in ('Eaten', 'Win', 'Terminal'):
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
        if path[i] not in ('Eaten', 'Win', 'Terminal'):
            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player
            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
        display.display(fig)
        time.sleep(0.1)
        display.clear_output(wait = True)


#############################################################


def learn_q(learning_func, env, initial_q, eps, delta, alpha, episodes):
    gamma= 49/50

    q_values = initial_q
    visits = np.ones((env.n_states, env.n_actions))

    start  = ((0,0), (6,5), 0)

    start_pos_value = np.zeros(episodes)
    for i in range(episodes):
        q_values, visits = learning_func(
            env,
            start,
            alpha,
            q_values,
            gamma,
            eps if delta==0 else 1/(i+1)**delta,
            visits
        )
        start_pos_value[i] = np.max(q_values[env.map[start], :])

        if i % 5000 == 0:
            print("episode iter ", i)

    policy = np.argmax(q_values, axis=1)

    return q_values, policy, start_pos_value

#############################################################

def init_Q(env, maze):
    key_cords = np.argwhere(maze == 3)[0]#(0, 7)
    exit_cords = np.argwhere(maze == 2)[0]

    Q_init = np.zeros((env.n_states, env.n_actions))

    for i in range(env.n_states-3):
        (player_cords, minotaur_cords, key) = env.states[i]
        if key == 0:
            Q_init[i,:] = 0.5*(1-np.linalg.norm(np.array(player_cords) - np.array(key_cords), ord=1)/15)


    for i in range(env.n_states-3):
        (player_cords, minotaur_cords, key) = env.states[i]
        if key == 1:
            Q_init[i,:] = 1-np.linalg.norm(np.array(player_cords) - np.array(exit_cords), ord=1)/30

    Q_init[env.map["Win"],:] = 1
    
    Q_init[:,env.STAY] = 0

    return Q_init

#############################################################

def run_training(eps, delta, alpha, episodes, method, learning_func, initial_q, save_files=False):
    name = f"{method}_eps_{eps:.2f}_delta_{delta:.2f}_alpha_{alpha:.2f}_init_{0 if np.sum(initial_q)==0 else 1}_episodes_{episodes}"
    print(name)

    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze

    env = Maze(maze) # Create an environment maze
    q_value, q_policy, start_pos_value = learn_q(learning_func, env, initial_q, eps, delta, alpha, episodes)

    # PLOTTING
    plt.clf()
    plt.plot(start_pos_value)
    plt.xlabel("episode")
    plt.ylabel("$V(start)$")

    title = f"Value at start using {method.upper()} with" +\
        (f" $\\varepsilon={eps:.2f}$" if delta==0 else f" $\\delta={delta:.2f}$") +\
        f",\n$\\alpha={alpha:.2f}$" +\
        f", and {"zero" if np.sum(initial_q)==0 else "informed"} initialization."

    plt.title(title)
    plt.ylim()
    plt.grid()
    if save_files:
        plt.savefig(f"graphs/{name}.png")
    else:
        plt.show()

    # SURVIVAL
    start  = ((0,0), (6,5), 0)
    n = 10000
    survives = 0
    for i in range(0,n):
        path, horizon = env.simulate(start, q_policy, method)
        if "Win" in path:
            survives += 1
    survival_prob = survives/n

    # SAVING
    if save_files:
        np.save(f"data/{name}.npy", {
            "q_value": q_value,
            "q_policy": q_policy,
            "start_pos_value": start_pos_value,
            "survival_prob": survival_prob
        })

    print(name, "Survival probability: " ,survival_prob)
    print()

###########################################################

if __name__ == "__main__":
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze
    env = Maze(maze) # Create an environment maze

    params = {
        "eps": 0.1, # exploration rate. Used if delta is 0
        "delta": 0, # exploration rate decay. Used if nonzero
        "alpha": 2/3, # learning rate decay
        "initial_q": init_Q(env, maze), # initialization of Q-values
        "method": "Q_learning", # "sarsa" or "Q_learning"
        "learning_func": q_learning, # sarsa or q_learning
        "episodes": 500 # number of episodes to run
    }

    run_training(**params)
