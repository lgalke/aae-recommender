
# torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from .base import Recommender

USE_CUDA = True

def RNRecommender(Recommender):
    def __init__(self,
                 embedding_dim=300,
                 hidden_size=256,
                 batch_size=16,
                 lr=1e-3):
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.item_embedding = None
        self.fcout = None

    def __str__(self):
        desc = "RelationNet Recommender"
        return desc

    def _encode_users(their_items):
        return self.item_embedding(their_items).mean(1)

    def _collate_fn(examples):
        maxlen = max(len(ex) for ex in examples)
        batch = [ex + [self.padding_idx] * (maxlen - len(ex))
                 for ex in examples]
        return torch.LongTensor(batch)

    def _lists2long(batch_items, max_items):
        out = torch.zeros(batch_items.size(0), max_items).long()
        ind = [(r, c) for r, cs in enumerate(input) for c in cs]
        if not ind:
            ind = ((), ())
        else:
            ind = tuple(zip(*ind))
        out[ind] = 1
        return out

    def train(train_set):
        n_users, n_items = train_set.size()
        self.padding_idx = n_items
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim,
                                           padding_idx=n_items)

        loader = torch.utils.DataLoader(
            train_set,
            collate_fn=self._collate_fn,
            pin_memory=True,
            shuffle=True,
            batch_size=self.batch_size
        )

        optimizer = optim.Adam(self.item_embedding.parameters(), lr=self.lr)

        self.fcout = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_size),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, n_items),
            nn.Sigmoid()
        )
        if CUDA:
            self.item_embedding = self.item_embedding.cuda()
            self.fcout = self.fcout.cuda()

        criterion = nn.BCELoss()

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            if USE_CUDA:
                batch = batch.cuda()
            batch = Variable(batch)
            user_embedding = self._encode_users(batch)
            out = self.fcout(user_embedding)

            loss = criterion(out, self._lists2long(batch))
            loss.backward()
            optimizer.step()
            print("[%d] %.4f" % (step, loss.item()))

    def predict(test_set):
        loader = torch.utils.DataLoader(
            test_set,
            collate_fn=self._collate_fn,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size
        )
        results = []

        with torch.no_grad():
            for batch in enumerate(loader):
                if USE_CUDA:
                    batch = batch.cuda()
                batch = Variable(batch)
                user_embedding = self._encode_users(batch)
                out = self.fcout(user_embedding)
                # warn: no sigmoid here, since in fcout
                results.append(out.data.cpu().numpy())

        return np.vstack(results)



