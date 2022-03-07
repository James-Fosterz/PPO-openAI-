import torch
import torch.nn as nn

torch.set_default_tensor_type('torch.FloatTensor')

#  Actor network class
class Actor(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate, dropout = 0.1, momentum = 0.9): 
        super(Actor, self).__init__()
    
        # Neural network structure
        '''
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
            #nn.Softmax(dim = -1)
            ) 
        '''        
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Cuda was not working and returned many issues so we ran soley on the cpu which was not ideal
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print(self.device)
        self.to(self.device)
       
    # Forward pass through actor network
    def forward(self, action):
        action = self.actor(action)     
        return action
        


# Critic network class
class Critic(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate, dropout = 0.1, momentum = 0.9):
        super(Critic, self).__init__()
        
        # Neural netowkr structure
        '''
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
            )
        '''      
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Cuda was not working and returned many issues so we ran soley on the cpu which was not ideal
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.to(self.device)
    
    # Forward pass through critic network
    def forward(self, value):       
        value = self.critic(value)      
        return value

'''
Momentum was only used when experimenting with other gradient ascents during testing as we tried to get our network to learn
We also learnt the original network was far to small so we increased the strcuture to that of someone elses
Lastly we moved Softmax from here and calulated it when needed inside of PPO
'''