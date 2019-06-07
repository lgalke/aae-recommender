import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05,
                 conditions=None):
        super(Generator, self).__init__()

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

        self.g_params = []

        if self.param == None:
            self.G_user_embeddings = Variable(
                torch.FloatTensor(self.userNum, self.emb_dim).uniform_(-self.initdelta, self.initdelta),
                requires_grad=True)
            self.G_item_embeddings = Variable(
                torch.FloatTensor(self.itemNum, self.emb_dim).uniform_(-self.initdelta, self.initdelta),
                requires_grad=True)
            self.G_item_bias = Variable(torch.zeros(self.itemNum, dtype=torch.float32), requires_grad=True)
        else:
            self.G_user_embeddings = torch.autograd.Variable(torch.tensor(param[0]).cuda(), requires_grad=True)
            self.G_item_embeddings = torch.autograd.Variable(torch.tensor(param[1]).cuda(), requires_grad=True)
            self.G_item_bias = torch.autograd.Variable(torch.tensor(param[2]).cuda(), requires_grad=True)

        self.g_params = [self.G_user_embeddings, self.G_item_embeddings, self.G_item_bias]
        self.optimizer = torch.optim.SGD(self.g_params, lr=self.learning_rate, momentum=0.9)

        if torch.cuda.is_available():
            self.G_user_embeddings = self.G_user_embeddings.cuda()
            self.G_item_embeddings = self.G_item_embeddings.cuda()
            self.G_item_bias = self.G_item_bias.cuda()

    def all_rating(self, user_pos, condition_data=None, impose_dim=None):
        # u_embedding = self.G_user_embeddings[user_index, :]
        if impose_dim == None or impose_dim == 0:
            u_embedding = torch.zeros(self.emb_dim, dtype=torch.float32)
            for i in user_pos:
                u_embedding.add(self.G_item_embeddings[i])
            u_embedding /= len(user_pos)
        else:
            u_embedding = torch.zeros([len(user_pos), self.emb_dim], dtype=torch.float32)
            for u in user_pos:
                for i in u:
                    u_embedding[u].add(self.G_item_embeddings[i])
                u_embedding[u] /= len(user_pos)
        item_embeddings = self.G_item_embeddings

        if self.conditions:
            u_embedding = self.conditions.encode_impose(u_embedding, condition_data, dim=impose_dim)
            u_embedding = self.lin(u_embedding)

        all_rating = torch.mm(u_embedding.view(-1, 5), item_embeddings.t()) + self.G_item_bias
        return all_rating

    def all_logits(self, user_pos, condition_data=None):
        # u_embedding = self.G_user_embeddings[user_index]
        u_embedding = torch.zeros(self.emb_dim, dtype=torch.float32)
        for i in user_pos:
            u_embedding.add(self.G_item_embeddings[i])
        u_embedding /= len(user_pos)

        if self.conditions:
            u_embedding = self.conditions.encode_impose(u_embedding, condition_data)
            u_embedding = self.lin(u_embedding)
        item_embeddings = self.G_item_embeddings

        score = torch.sum(u_embedding*item_embeddings, 1) + self.G_item_bias
        return score

    def forward(self, user_index, sample, reward, condition_data=None):
        softmax_score = F.softmax(self.all_logits(user_index, condition_data).view(1, -1), -1)
        gan_prob = torch.gather(softmax_score.view(-1), 0, sample.long()).clamp(min=1e-8)
        loss = -torch.mean(torch.log(gan_prob) * reward)

        return loss

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
