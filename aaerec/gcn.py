""" Graph convolutions 
Code derived https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
"""

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

from .base import Recommender

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}


class GCN(nn.Module):
    """ A Graph convolutional layer """
    # TODO FIXME dropout omitted
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    """ A net of graph convolutional layers """
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x


class GCNRecommender(Recommender):
    """ A recommender based on graph convolutions """
    def __init__(self, embedding_dim, hidden_dim, code_dim):
        super(GCNRecommender, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim


    def train(self, x_train):
        # TODO open problem: we don't have users of test set during training
        pass


    def predict(self, x_test):
        # TODO open problem: we don't have users of test set during training
        pass

