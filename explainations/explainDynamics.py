#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
# from torch_geometric.nn import GCNConv, Set2Set, GNNExplainer
model = torch.load("./model_1040000.pth")


# In[2]:


from gn_models import init_graph_features, FFGN
node_feat_size = 6
edge_feat_size = 3
graph_feat_size = 10
gn = FFGN(graph_feat_size, node_feat_size, edge_feat_size).cuda()
gn.load_state_dict(model)
# explainer = GNNExplainer(gn, epochs=200)


# In[3]:


from train_gn import SwimmerDataset
from torch.utils.data import DataLoader
import networkx as nx
import numpy as np
import matplotlib.pyplot as plot
from evaluate_gn import evaluate_graph_loss

get_ipython().run_line_magic('matplotlib', 'inline')
threshold = 60
dset = SwimmerDataset('swimmer2.npy')
dl = DataLoader(dset, batch_size=200, num_workers=0, drop_last=True)
G1 = nx.path_graph(6).to_directed()
normalizers = torch.load('normalize.pth')
in_normalizer = normalizers['in_normalizer']
out_normalizer = normalizers['out_normalizer']
std = in_normalizer.get_std()
from utils import load_graph_features
for i,data in enumerate(dl):
    action, delta_state, last_state = data
    action, delta_state, last_state = action.float(), delta_state.float(), last_state.float()
    if torch.cuda.is_available():
        action, delta_state, last_state = action.cuda(), delta_state.cuda(), last_state.cuda()
    dpos = delta_state[:, 5:5 + 18].view(-1, 6, 3)
    last_pos = last_state[:, 5:5 + 18].view(-1, 6, 3)
    init_graph_features(G1, graph_feat_size, node_feat_size, edge_feat_size, cuda=True, bs = 200)
    load_graph_features(G1, action, last_state, delta_state, bs=200, noise = 0.03, std = std)
    G_out = gn(in_normalizer.normalize(G1))
    G_out = out_normalizer.inormalize(G_out)
    loss, true, pred = evaluate_graph_loss(G_out, delta_state, last_state)
    nodes =[]
    # plot.figure()
    for node in G_out.nodes():
        pred = G_out.nodes[node]['feat'][:,:3]
        predpos = G_out.nodes[node]['feat'][0,:3].cpu().data.numpy() + last_pos[0,node,:3].cpu().data.numpy()
        true = dpos[:,node]
        if (torch.mean(torch.abs((pred - true) / true )).item()) > threshold:
            nodes.append(node)
    print(nodes)
    dictionary = {}
    for i in range(200):
        for j in G_out.nodes:
            if i not in dictionary.keys():
                dictionary[i] = {}
            if j not in dictionary[i].keys():
                dictionary[i][j] = []
            dictionary[i][j].append(G_out.nodes[j]['feat'][i].cpu().detach().numpy())


# In[4]:


#understanding time based decay of features
array2d = []
for i in dictionary.keys():
    arrt = []
    for j in dictionary[i].keys():
        arrt.append(dictionary[i][j])
    array2d.append(arrt)
print(len(array2d[0][0]))
# arr3 = []
# for i in dictionary.keys():
#     arr3.append(array2d
array3d= np.array(array2d).reshape(200,6,6)
new2d = array3d.transpose(1,0,2).reshape(6,-1)
v_min = array3d.min(axis=( 0), keepdims=True)
v_max = array3d.max(axis=( 0), keepdims=True)
# print(v_min.shape)
# print(v_max.shape)
# # array3d = np.dstack([[[x for x in dictionary[i][j]] for j in dictionary[i].keys()] for i in dictionary.keys()])
# print(array3d.shape)
# print(array3d[0])


# In[5]:


sim = new2d.dot(new2d.T)


# In[6]:


#exploiting induced bias to make explicite understanding of the system
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

ax = sns.heatmap(sim, linewidth=0.5,cmap="PiYG")
plt.show()

