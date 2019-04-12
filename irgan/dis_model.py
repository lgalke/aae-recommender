import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from .utils import L2Loss


class Discriminator(nn.Module):
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05,
                 conditions=None):
        super(Discriminator, self).__init__()

        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.conditions = conditions
        if self.conditions:
            self.lin = nn.Linear(self.emb_dim + self.conditions.size_increment(), self.emb_dim)

        self.d_param = []

        if self.param == None:
            self.D_user_embeddings = Variable(
                torch.FloatTensor(self.userNum, self.emb_dim).uniform_(-self.initdelta, self.initdelta),
                requires_grad=True)
            self.D_item_embeddings = Variable(
                torch.FloatTensor(self.itemNum, self.emb_dim).uniform_(-self.initdelta, self.initdelta),
                requires_grad=True)
            self.D_item_bias = Variable(torch.zeros(self.itemNum, dtype=torch.float32), requires_grad=True)
        else:
            self.D_user_embeddings = torch.autograd.Variable(torch.tensor(param[0]).cuda(), requires_grad=True)
            self.D_item_embeddings = torch.autograd.Variable(torch.tensor(param[1]).cuda(), requires_grad=True)
            self.D_item_bias = torch.autograd.Variable(torch.tensor(param[2]).cuda(), requires_grad=True)

        self.d_param = [self.D_user_embeddings, self.D_item_embeddings, self.D_item_bias]
        self.optimizer = torch.optim.SGD(self.d_param, lr=self.learning_rate, momentum=0.9)
        self.l2l = L2Loss()
        if torch.cuda.is_available():
            self.D_user_embeddings = self.D_user_embeddings.cuda()
            self.D_item_embeddings = self.D_item_embeddings.cuda()
            self.D_item_bias = self.D_item_bias.cuda()
            self.l2l = self.l2l.cuda()

    def pre_logits(self, input_user, input_item, condition_data=None):
        u_embedding = self.D_user_embeddings[input_user, :]
        if self.conditions:
            # In generator need to use dimension 0 in discriminator 1 so by default 0 (given in condition creation)
            # and here we change impose and set it back to 0
            # TODO Better solutionis to always use a batch instead of a specific user as for all other methods
            for c in self.conditions:
                self.conditions[c].dim = 1

            print(u_embedding.size(), torch.Tensor(condition_data).size())
            u_embedding = self.conditions.encode_impose(u_embedding, condition_data)

            for c in self.conditions:
                self.conditions[c].dim = 0

            u_embedding = self.lin(u_embedding)

        item_embeddings = self.D_item_embeddings[input_item, :]
        D_item_bias = self.D_item_bias[input_item]

        score = torch.sum(u_embedding*item_embeddings, 1) + D_item_bias
        return score

    def forward(self, input_user, input_item, pred_data_label, condition_data=None):
        loss = F.binary_cross_entropy_with_logits(self.pre_logits(input_user, input_item, condition_data), pred_data_label.float()) \
            + self.lamda * (self.l2l(self.D_user_embeddings) + self.l2l(self.D_item_embeddings) + self.l2l(self.D_item_bias))
        return loss

    def get_reward(self, user_index, sample):
        u_embedding = self.D_user_embeddings[user_index, :]
        item_embeddings = self.D_item_embeddings[sample, :]
        D_item_bias = self.D_item_bias[sample]

        reward_logits = torch.sum(u_embedding*item_embeddings, 1) + D_item_bias
        reward = 2 * (torch.sigmoid(reward_logits) - 0.5)
        return reward

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

