import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim

class Grid:
    def __init__(self, d, l, spawn_pos, end_pos, eps=0, closed=True):
        self.d = d
        self.l = l
        self.spawn_pos = spawn_pos
        self.end_pos = end_pos
        self.closed = closed
        self.eps = eps

        self.pos = spawn_pos

    def step(self, action):

        # Env randomness
        if np.random.rand() < self.eps:
            dir = np.random.randint(0, self.d)
            sign = np.random.randint(0,2)
        else:
            assert -self.d <= action <= self.d and action != 0
            dir = abs(action)-1
            if action > 0:
                sign = 1
            else:
                sign = -1

        # Changing position
        if self.closed:
            self.pos[dir] = min(self.l-1, max(self.pos[dir] + sign, 0))
        else:
            self.pos[dir] = (self.pos + sign )% self.l

        # Return status
        if self.pos == self.end_pos:
            return self.pos.copy(), True
        else:
            return self.pos.copy(), False

    def reset(self):
        self.pos = self.spawn_pos
        return self.pos.copy()


def render(positions, l):
    assert positions.shape[1] == 2
    plt.scatter(positions[:, 0], positions[:, 1], c=range(positions.shape[0]))
    plt.xlim(-1, l)
    plt.ylim(-1, l)
    plt.show()

class FiniteMDP:
#Finite MDP with a set of Actions A available in every state.   
    def __init__(self,Ns,Na,P,R,start_state=None, final_states=[]):
        self.Ns = Ns
        self.Na = Na
        assert P.shape == (Na,Ns,Ns)
        self.P = P
        assert R.shape == (Ns,Na)
        self.R = R
        if start_state is None:
            #init at a random state
            self.state = torch.multinomial(torch.ones(Ns),1).item()
        else:
            assert 0 <= start_state <= self.Ns
            self.state = start_state
        self.final_states = final_states
    
    def step(self,action):
        reward = self.R[action,state]
        next_state = torch.multinomial(self.P[action,self.state],1).item()
        self.state = next_state
        done =False
        if self.state in self.final_states:
            done =True
        return self.state.copy(), reward, done
    
    def reset(self,start_state=None):
        self.state = torch.multinomial(torch.ones(Ns),1).item()
        if not (start_state is None):
            self.state = start_state
        return self.state
    
    def render(self): #Plot the graph corresponding to the markov chain defined by a uniform random walker
        G=nx.Graph()
        G.add_nodes_from(range(self.Ns))
        rw_matrix = torch.mean(self.P,dim=0)
        edges = torch.where(rw_matrix > 0)
        edges_list=[]
        for i in range(edges[0].shape[0]):
            s = edges[0][i].item()
            s_prime = edges[1][i].item()
            w = rw_matrix[s][s_prime].item()
            if s <= s_prime:
                edges_list.append((s,s_prime,w))
        G.add_weighted_edges_from(edges_list)
        colors = range(len(edges_list))
        options = {
            "edge_color": colors,
            "width": 30./self.Ns,
            "edge_cmap": plt.cm.Blues,
            "with_labels": True,
        }
        nx.draw(G,**options)
        plt.savefig("Finite_MDP_graph.png") # save as png
        plt.show() # display

def Grid_to_MDP(env):    
    l = env.l
    d = env.d
    Ns = l**d
    Na = 2*d
    idx = 0
    index = {}
    states = {}
    for state in np.ndindex(tuple([l] * d)):
        index[state] = idx
        states[idx] = state
        idx+=1 
    P = torch.zeros(Na,Ns,Ns)
    for state in np.ndindex(tuple([l] * d)):
        #actions are indexed as follows [LEFT,RIGHT,UP,DOWN]
        #TODO : generalize to case d > 2
            left_s = (state[0]-1,state[1])
            right_s = (state[0]+1,state[1])
            up_s = (state[0],state[1]+1)
            down_s = (state[0],state[1]-1)
            for a,state_prime in enumerate([left_s,right_s,up_s,down_s]):
                try: 
                    P[a,index[state],index[state_prime]] = 1
                except :
                    #if state is on the border, doing action a keeps agent at same place.
                    P[a,index[state],index[state]] = 1
    R = torch.zeros(Ns,Na)
    mdp = FiniteMDP(Ns,Na,P,R)
    return mdp,states