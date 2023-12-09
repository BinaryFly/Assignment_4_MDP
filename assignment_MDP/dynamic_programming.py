#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from world import World

class Dynamic_Programming:

    def __init__(self):
        self.V_s = None # will store a potential value solution table
        self.Q_sa = None # will store a potential action-value solution table
        
    def value_iteration(self,env: World,gamma = 1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Value Iteration (VI)")
        # initialize value table
        V_s = np.zeros(env.n_states)

        delta = theta * 2   # make sure delta is more than theta the first iteration
        while delta >= theta:
            delta = 0
            for state in env.states:
                old_value = V_s[state]
                highest = None
                for action in env.actions:
                    new_state, return_value = env.transition_function(state, action)
                    calculated_value = float(return_value + gamma * V_s[new_state])  
                    if highest is None or highest < calculated_value:
                        highest = calculated_value
                    # highest = calculated_value if highest is None or highest < calculated_value else highest
                V_s[state] = highest

                # this is just to satisfy the linter
                if highest is None:
                    continue

                delta = max([delta, float(old_value) - highest])
            print(f"error: {delta}")

        self.V_s = V_s
        return

    def Q_value_iteration(self,env: World,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        delta = theta * 2   # make sure delta is more than theta the first iteration
        while delta >= theta:
            delta = 0
            for state in env.states:
                for index, action in enumerate(env.actions):
                    old_value = Q_sa[state][index]
                    new_state, return_value = env.transition_function(state, action)

                    # get the maximum return value for the next state after transitioning 
                    max_next_return_value = Q_sa[new_state].max()
                    calculated_value = float(return_value + gamma * max_next_return_value)  
                    Q_sa[state][index] = calculated_value
                    delta = max([delta, float(old_value) - calculated_value])

            print(f"error: {delta}")

        ## IMPLEMENT YOUR Q-VALUE ITERATION ALGORITHM HERE

        self.Q_sa = Q_sa
        return
                
    def execute_policy(self,env: World,table='V'):
        ## Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state() # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None:
                ## IMPLEMENT ACTION VALUE ESTIMATION FROM self.V_s HERE !!!
                # print("You still need to implement greedy action selection from the value table self.V_s!")
                greedy_action = None
                best_value = None
                for action in env.actions:
                    new_state, return_value = env.transition_function(current_state, action)
                    value = self.V_s[new_state]
                    if return_value > 0: # this is the goal state
                        greedy_action = action
                        break
                    if best_value is None or float(value) > best_value:
                        greedy_action = action
                        best_value = float(value)
            
            elif table == 'Q' and self.Q_sa is not None:
                ## IMPLEMENT ACTION VALUE ESTIMATION FROM self.Q_sa here !!!
                
                greedy_action = None
                best_value = self.Q_sa[current_state][0]
                max_action_value_index = 0
                for index, action_value in enumerate(self.Q_sa[current_state]):
                    if env.transition_function(current_state,env.actions[index])[1] > 0:
                        max_action_value_index = index
                        break
                    
                    if action_value > best_value:
                        max_action_value_index = index
                        best_value = action_value
                        
                greedy_action = env.actions[max_action_value_index]                  
                
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None


            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
    
if __name__ == '__main__':
    env = World('prison.txt') 
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy
    
    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy

