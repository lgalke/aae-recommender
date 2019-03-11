from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import sklearn
from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import scipy.sparse as sp

try:
# within module
    from .condition import _check_conditions, ConditionList, PretrainedWordEmbeddingCondition
    from .base import Recommender
    from .datasets import Bags
    from .evaluation import Evaluation
except SystemError:
# if executable
    from aaerec.condition import _check_conditions, ConditionList, PretrainedWordEmbeddingCondition
    from aaerec.base import Recommender
    from aaerec.datasets import Bags
    from aaerec.evaluation import Evaluation

torch.manual_seed(42)

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True

TORCH_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}


class VAE(nn.Module):

    def __init__(self,
                 inp,
                 out,
                 n_hidden=100,
                 n_code=50,
                 lr=0.001,
                 batch_size=100,
                 n_epochs=500,
                 optimizer='adam',
                 normalize_inputs=True,
                 activation='ReLU',
                 final_activation='Sigmoid',
                 # TODO try later
                 # dropout=(.2,.2),
                 conditions=None,
                 verbose=True,
                 log_interval=1):

        super(VAE, self).__init__()

        self.normalize_inputs = normalize_inputs
        self.inp = inp
        self.n_hidden = n_hidden
        self.n_code = n_code
        self.n_epochs = n_epochs
        self.verbose = verbose
        # TODO try later
        # self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.activation = activation
        self.conditions = conditions

        # 2-layers Encoder
        self.fc1 = nn.Linear(inp, n_hidden)
        self.fc21 = nn.Linear(n_hidden, n_code)
        self.fc22 = nn.Linear(n_hidden, n_code)

        # 2-layers Decoder
        if self.conditions:
            n_code += self.conditions.size_increment()
        self.fc3 = nn.Linear(n_code, n_hidden)
        self.fc4 = nn.Linear(n_hidden, out)
        optimizer_gen = TORCH_OPTIMIZERS[optimizer.lower()]
        self.optimizer = optimizer_gen(self.parameters(), lr=lr)

        self.act = getattr(nn, activation)()
        self.final_act = getattr(nn, final_activation)()

        self.log_interval = log_interval

    def encode(self, x):
        h1 = self.act(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.act(self.fc3(z))
        return self.final_act(self.fc4(h3))

    def forward(self, x, condition_data=None):
        if self.normalize_inputs:
            x = F.normalize(x, 1)
        # TODO could I use x instead of self.inp? Do we need x.view?
        mu, logvar = self.encode(x.view(-1, self.inp))
        z = self.reparametrize(mu, logvar)
        use_condition = _check_conditions(self.conditions, condition_data)
        if use_condition:
            z = self.conditions.encode_impose(z, condition_data)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        reconstruction_function = nn.BCELoss()
        reconstruction_function.size_average = False

        BCE = reconstruction_function(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return BCE + KLD

    def partial_fit(self, X, y=None, condition_data=None):
        """
            Performs reconstrction, discimination, generator training steps
        :param X: np.array, the base data from Bag class
        :param y: dummy variable, throws Error if used
        :param condition_data: generic list of conditions
        :return:
        """
        ### DONE Adapt to generic condition ###
        use_condition = _check_conditions(self.conditions, condition_data)

        if y is not None:
            raise ValueError("(Semi-)supervised usage not supported")
        # Transform to Torch (Cuda) Variable, shift batch to GPU
        X = Variable(torch.FloatTensor(X))
        if torch.cuda.is_available():
            X = X.cuda()

        # Make sure we are in training mode and zero leftover gradients
        self.train()
        train_loss = 0
        self.optimizer.zero_grad()
        # Build the model on the concatenated data, compute BCE without concatenation
        if use_condition:
            recon_batch, mu, logvar = self(X, condition_data)
        else:
            recon_batch, mu, logvar = self(X)

        loss = self.loss_function(recon_batch, X, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        self.optimizer.step()
        # if self.verbose and batch_idx % self.log_interval == 0:
        #     print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader),
        #                loss.data[0] / len(data)))
        if self.verbose:
            print('====> Average loss: {:.4f}'.format(
                train_loss / len(X)))
        return self

    def fit(self, X, y=None, condition_data=None):
        """
        :param X: np.array, the base data from Bag class
        :param y: dummy variable, throws Error if used
        :param condition_data: generic list of conditions
        :return:
        """
        ### DONE Adapt to generic condition ###
        # TODO: check how X representation and numpy.array work together
        # TODO: adapt combining X and new_conditions_name
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")

        use_condition = _check_conditions(self.conditions, condition_data)

        # do the actual training
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("Epoch", epoch + 1)

            if use_condition:
                # shuffle(*arrays) takes several arrays and shuffles them so indices are still matching
                X_shuf, *condition_data_shuf = sklearn.utils.shuffle(X, *condition_data)
            else:
                X_shuf = sklearn.utils.shuffle(X)

            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuf[start:end].toarray()
                # condition may be None
                if use_condition:
                    # c_batch = condition_shuf[start:(start+self.batch_size)]
                    c_batch = [c[start:end] for c in condition_data_shuf]
                    self.partial_fit(X_batch, condition_data=c_batch)
                else:
                    self.partial_fit(X_batch)

            if self.verbose:
                # Clean up after flushing batch loss printings
                print()
        return self

    def predict(self, X, condition_data=None):
        """
        :param X: np.array, the base data from Bag class
        :param condition_data: generic list of conditions
        :return:
        """
        ### DONE Adapt to generic condition ###
        use_condition = _check_conditions(self.conditions, condition_data)

        self.eval()  # Deactivate dropout
        if self.conditions:
            self.conditions.eval()
        pred = []
        for start in range(0, X.shape[0], self.batch_size):
            # batched predictions, yet inclusive
            end = start + self.batch_size
            X_batch = X[start:end].toarray()
            if sp.issparse(X_batch):
                X_batch = X_batch.toarray()
            X_batch = Variable(torch.FloatTensor(X_batch), volatile=True)
            if torch.cuda.is_available():
                X_batch = X_batch.cuda()

            if use_condition:
                c_batch = [c[start:end] for c in condition_data]

            test_loss = 0

            if use_condition:
                recon_batch, mu, logvar = self(self.conditions.encode_impose(X_batch, c_batch))
            else:
                recon_batch, mu, logvar = self(X_batch)
            test_loss += self.loss_function(recon_batch, X_batch, mu, logvar).data[0]
            pred.append(recon_batch.data.cpu().numpy())

        test_loss /= len(X_batch)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        return np.vstack(pred)


class VAERecommender(Recommender):
    """
    Varietional Autoencoder Recommender
    =====================================

    Arguments
    ---------
    n_input: Dimension of input to expect
    n_hidden: Dimension for hidden layers
    n_code: Code Dimension

    Keyword Arguments
    -----------------
    n_epochs: Number of epochs to train
    batch_size: Batch size to use for training
    verbose: Print losses during training
    normalize_inputs: Whether l1-normalization is performed on the input
    """
    def __init__(self, conditions=None,
                 **kwargs):

        super().__init__()
        self.verbose = kwargs.get('verbose', True)
        self.conditions = conditions
        self.model_params = kwargs
        self.model = None

    def __str__(self):
        desc = "Variational Autoencoder"
        if self.conditions:
            desc += " conditioned on: " + ', '.join(self.conditions.keys())
        desc += '\nModel Params: ' + str(self.model_params)
        return desc

    def train(self, training_set):
        ### DONE Adapt to generic condition ###
        """
        1. get basic representation
        2. ? add potential side_info in ??? representation
        3. initialize a Variational Autoencoder
        4. fit based on Variational Autoencoder
        :param training_set: ???, Bag Class training set
        :return: trained self
        """
        X = training_set.tocsr()
        if self.conditions:
            condition_data_raw = training_set.get_attributes(self.conditions.keys())
            condition_data = self.conditions.fit_transform(condition_data_raw)
            #self.model = VAE(X.shape[1] + self.conditions.size_increment(), X.shape[1],
            #                 conditions=self.conditions, **self.model_params)
        else:
            condition_data = None
            #self.model = VAE(X.shape[1], X.shape[1], **self.model_params)
        self.model = VAE(X.shape[1], X.shape[1], conditions=self.conditions, **self.model_params)

        print(self)
        print(self.model)
        print(self.conditions)

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.fit(X, condition_data=condition_data)

    def predict(self, test_set):
        X = test_set.tocsr()
        if self.conditions:
            condition_data_raw = test_set.get_attributes(self.conditions.keys())
            # Important to not call fit here, but just transform
            condition_data = self.conditions.transform(condition_data_raw)
        else:
            condition_data = None

        pred = self.model.predict(X, condition_data=condition_data)

        return pred


def main():
    """ Evaluates the VAE Recommender """
    CONFIG = {
        'pub': ('/data21/lgalke/datasets/citations_pmc.tsv', 2011, 50),
        'eco': ('/data21/lgalke/datasets/econbiz62k.tsv', 2012, 1)
    }

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('data', type=str, choices=['pub','eco'])
    args = PARSER.parse_args()
    DATA = CONFIG[args.data]
    logfile = '/data22/ivagliano/test-vae/' + args.data + '-decoder.log'
    bags = Bags.load_tabcomma_format(DATA[0])
    c_year = DATA[1]

    evaluate = Evaluation(bags,
                          year=c_year,
                          logfile=logfile).setup(min_count=DATA[2],
                                                 min_elements=2)
    print("Loading pre-trained embedding", W2V_PATH)
    vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

    params = {
        'n_epochs': 10,
        'batch_size': 100,
        'optimizer': 'adam',
        # 'normalize_inputs': True,
    }

    CONDITIONS = ConditionList([
        ('title', PretrainedWordEmbeddingCondition(vectors))
    ])

    # 100 hidden units, 200 epochs, bernoulli prior, normalized inputs -> 0.174
    # activations = ['ReLU','SELU']
    # lrs = [(0.001, 0.0005), (0.001, 0.001)]
    # hcs = [(100, 50), (300, 100)]

    # dropouts = [(.2,.2), (.1,.1), (.1, .2), (.25, .25), (.3,.3)] # .2,.2 is best
    # priors = ['categorical'] # gauss is best
    # normal = [True, False]
    # bernoulli was good, letz see if categorical is better... No
    # import itertools
    # models = [VAERecommender(**params, n_hidden=hc[0], n_code=hc[1],
    #                          use_title=ut, embedding=vectors,
    #                          gen_lr=lr[0], reg_lr=lr[1], activation=a)
    #           for ut, lr, hc, a in itertools.product((True, False), lrs, hcs, activations)]
    models = [VAERecommender(conditions=CONDITIONS, **params)]
    evaluate(models)


if __name__ == '__main__':
    main()
