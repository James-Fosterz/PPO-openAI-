import torch
import gym
import numpy as np

import torch.distributions as distributions
import torch.nn.functional as F

from ActorCritic import Actor
from ActorCritic import Critic

import matplotlib.pyplot as plt


# Proximal Policy Class
class PPO:
    # Initialise the class networks and hyperparamteres
    def __init__(self, environment):
        # The Lunar Lander environment
        self.env = environment
        
        # Hyperparameters
        self.lr = 0.001
        self.discount = 0.99
        self.ppo_steps = 1
        self.ppo_clip = 0.5
        
        # Dimmensions for use in the Neual Networks
        # Input for both is the observation or state from enviornement
        self.input_dim = environment.observation_space.shape[0]
        self.hidden_dim = 128
        self.output_dim = environment.action_space.n
        
        # Intislises our two networks
        self.actor = Actor(self.input_dim, self.hidden_dim, self.output_dim, self.lr) # actor is for advantage calculation
        self.critic = Critic(self.input_dim, self.hidden_dim, 1, self.lr) # critic is for value calculation
        
        # Creates an optimiser for eahc network (these can also be pulled from The ActorCritic classes themselves)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Renders the enviorment while training so you can watch to lunar lunar attempt to land  (is slower with = True)
        self.render_env = False

        
    # The main loop
    def run(self):       
        # Trains the two networks
        self.actor.train()
        self.critic.train()
        
        # Resets the current obs to a new environment
        obs = self.env.reset()
        
        # Sets done to False as it is a new episode
        done = False
        # Score refers to the total reward recieceved from and episodes or game (sometime called return or episode reward)
        score = 0
        
        # Intisliases memory for all possible infoamriton we need to store
        # Reset at the start of every episode when this fucntion is run
        observations = []
        new_obseravtions = [] # No longer used in current calcultion foramt
        action_predictions = []
        action_probabilities = []
        actions = []
        rewards = []
        dones = [] # Would have been intersting to do more with the done boolean
        values = []
            
        # Runs each game until any of the games end critieria has been met
        while not done:
            
            # Renders the enviornement is this has been turned on
            if self.render_env:
                self.env.render()
            
            # The current observation is stored in memory
            obs = torch.FloatTensor(obs).unsqueeze(0)
            observations.append(obs)
                
            # Agent choses action based on the actor network
            # We return it in various forms for different uses
            # We also return the values here as well
            action_pred, action, action_prob, value_pred = self.get_actions(obs)
         
            # Perform the current action in the environment and retieves the information
            obs, reward, done, _ = self.env.step(action.item())                                  
            
            # Saves the relevant information obtained in the memory 
            actions.append(action)
            action_predictions.append(action_pred)
            action_probabilities.append(action_prob)
            rewards.append(reward)
            values.append(value_pred)       
            dones.append(done)
            
            # Add the reward for this step to the score (return) of this episode
            score += reward
        
        # Prepare the inforamtion for use in loss calculations and backprop
        observations = torch.cat(observations)
        actions = torch.cat(actions)    
        action_predictions = torch.cat(action_predictions)
        action_probabilities = torch.cat(action_probabilities)
        values = torch.cat(values).squeeze()
        
        # Calculates returns and advantages
        returns = self.calc_returns(rewards, self.discount)
        advantages = self.calc_advantages(returns, values)
        
        '''
        Originally we iterated episodes but due to our code structure and the nature of the game it worked best to run 1 episode per epoch
        Then perform training on the gathered inforamtion for that game
        Hyperparameters where then tuned accordingly
        '''
        
        # Calls the update fucntion where the network is trained and passes in all possible needed information
        policy_loss, value_loss = self.update(observations, actions, action_predictions, action_probabilities, advantages, returns)
     
        # Closes the environment if it was opened
        if self.render_env:
            self.env.close()
        
        # USED FOR DEBUGGING
        #print("finished")            
        
        # Returns the two loss vlaues and episodes score which is the vlaue we area aiming to maximise
        return policy_loss, value_loss, score 
        
    # Calculates returns
    def calc_returns(self, rewards, discount, normalize = True):
        
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + R * discount
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        
        if normalize:
            returns = (returns - returns.mean()) / returns.std()
        
        return returns
    
    
    # Calculates advantages
    def calc_advantages(self, returns, values, normalize = True):
        
        advantages = returns - values

        if normalize:
            advantages = (advantages - advantages.mean()) / advantages.std()
        
        return advantages
        
    
    # Update function where training of networks occurs
    def update(self, observations, actions, action_predictions, action_probabilities, advantages, returns):
        
        # USED FOR DEBUGGING
        #print("update")
        
        # Resets total loss to 0 for this episode
        total_policy_loss = 0
        total_value_loss = 0
        
        observations = observations.detach()
        actions = actions.detach()
        action_probabilities = action_probabilities.detach()
        advantages = advantages.detach()
        returns = returns.detach()
        
        # Choses how many steps of ppo to perform this for, this is an adjustable paramter should you wish
        for p in range(self.ppo_steps):
            
            action_pred = self.actor.forward(observations)
            value_pred = self.critic.forward(observations)
            value_pred = value_pred.squeeze(-1)
            value_pred = value_pred.flatten()
            action_prob = F.softmax(action_pred, dim = -1)
            dist = distributions.Categorical(action_prob)
            
            new_action_probabilities = dist.log_prob(actions)
            
            # Policy loss calculation via ppo_clip
            policy_ratio = (new_action_probabilities - action_probabilities).exp()      
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.ppo_clip, max = 1.0 + self.ppo_clip) * advantages        
            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()                 
            
            # Value loss caculation
            value_loss = F.smooth_l1_loss(returns, value_pred).mean()
            value_loss = torch.tensor(value_loss, dtype = float, requires_grad = True)
            
            # Performsstochastic gradient ascent using Adam and the loss values for each network
            self.actor.optim.zero_grad()
            policy_loss.backward(retain_graph = True)
            self.actor.optim.step()       
            self.critic.optim.zero_grad()
            value_loss.backward()
            self.critic.optim.step()
            
            # Adds current loss to total loss
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # USED FOR DEBUGGING
            #print('update done')
            
        return total_policy_loss / self.ppo_steps, total_value_loss / self.ppo_steps
            

    # Evaluates the models
    def evaluate(self):
        # This fuction is used to test the networks on other enviornemnts of Lunar Lander and see if it performs well on average
        self.actor.eval()
        self.critic.eval()
        
        #rewards = []
        done = False
        score = 0
        
        obs = self.env.reset()
        
        while not done:
            
            obs = torch.FloatTensor(obs).squeeze(0)
            
            _, action, _, _ = self.get_actions(obs)
            
            action = action.squeeze(-1)
            
            new_obs, reward, done, _ = self.env.step(action.item())
            
            score += reward
            
        return score
   
    # Gets an action from the actor network based on the current observation (state)
    # Generates an action prediction via the actor network adn returns it in various forms
    # Also generates a value prediciton via the critic network
    def get_actions(self, obs):
        
        action_pred = self.actor.forward(obs)
        
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        
        action_prob = dist.log_prob(action)
        
        value_pred = self.critic.forward(obs).to(self.critic.device)
        
        return action_pred, action, action_prob, value_pred
        
