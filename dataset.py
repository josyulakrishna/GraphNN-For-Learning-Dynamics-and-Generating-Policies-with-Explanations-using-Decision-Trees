import torch.utils.data as data
import numpy as np
import torch
from dm_control import suite


class SwimmerDataset(data.Dataset):
    def __init__(self, path, n):
        self.data = np.load(path)
        self.n = n-1
        if n == 3:
            self.env = suite.load(domain_name="myswimmer", task_name="swimmer")
        if n == 6:
            self.env = suite.load(domain_name="myswimmer", task_name="swimmer6")
        self.obs = self.env.observation_spec()

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - 2)

    def __getitem__(self, idx):
        episode = idx // (self.data.shape[1] - 2)
        frame = idx % (self.data.shape[1] - 2) + 1
        #print(episode, frame)

        last_state = self.data[episode, frame - 1,self.n:]
        this_state = self.data[episode, frame,self.n:]
        action = self.data[episode, frame, :self.n]
        l = self.obs["joints"].shape[0]+self.obs["abs"].shape[0]+self.obs["body_velocities"].shape[0]
        print(l)
        pos = last_state[self.n:self.n + l].reshape(6, 3)
        #pos += np.random.normal(scale = 0.001, size = pos.shape)
        last_state[self.n:self.n + l] = pos.reshape(18,)

        delta_state = this_state - last_state
        delta_state[delta_state > np.pi] -= np.pi * 2
        delta_state[delta_state < -np.pi] += np.pi * 2

        return action, delta_state, last_state
    
    
    def __get_episode__(self, idx):
        episode = idx 
        #print(episode, frame)
        l = self.obs["joints"].shape[0]+self.obs["abs"].shape[0]
        actions = []
        delta_states = []
        last_states = []
        
        for frame in range(10,110):
        
            last_state = self.data[episode, frame - 1,self.n:]
            this_state = self.data[episode, frame,self.n:]
            action = self.data[episode, frame, :self.n]

            pos = last_state[self.n:self.n + l].reshape(6, 3)
            #pos += np.random.normal(scale = 0.001, size = pos.shape)
            last_state[self.n:self.n + l] = pos.reshape(18,)

            delta_state = this_state - last_state
            delta_state[delta_state > np.pi] -= np.pi * 2
            delta_state[delta_state < -np.pi] += np.pi * 2

            actions.append(action)
            delta_states.append(delta_state)
            last_states.append(last_state)
        
        actions = np.array(actions)
        delta_states = np.array(delta_states)
        last_states = np.array(last_states)

        return actions, delta_states, last_states