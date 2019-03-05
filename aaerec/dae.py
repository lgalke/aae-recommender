""" Denoising Autoencoders """
# CFR https://gist.github.com/bigsnarfdude/dde651f6e06f266b48bc3750ac730f80,
# https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/tree/master/07_Denoising_Autoencoder

# torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable

# sklearn
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# numpy
import numpy as np
import scipy.sparse as sp

# own recommender stuff
from aaerec.base import Recommender
from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.ub import GensimEmbeddedVectorizer
from gensim.models.keyedvectors import KeyedVectors

torch.manual_seed(42)

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True

STATUS_FORMAT = "[ R: {:.4f} | D: {:.4f} | G: {:.4f} ]"


def log_losses(*losses):
    print('\r' + STATUS_FORMAT.format(*losses), end='', flush=True)


class Encoder(nn.Module):
    """ Three-layer Encoder """

    def __init__(self, n_input, n_hidden, n_code, final_activation=None,
                 normalize_inputs=True, dropout=(.2, .2), activation='ReLU'):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(n_input, n_hidden)
        self.act1 = getattr(nn, activation)()
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = getattr(nn, activation)()
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.lin3 = nn.Linear(n_hidden, n_code)
        self.normalize_inputs = normalize_inputs
        if final_activation == 'linear' or final_activation is None:
            self.final_activation = None
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError("Final activation unknown:", activation)

    def forward(self, inp):
        """ Forward method implementation of 3-layer encoder """
        if self.normalize_inputs:
            inp = F.normalize(inp, 1)
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # third layer
        act = self.lin3(act)
        if self.final_activation:
            act = self.final_activation(act)
        return act


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, n_code, n_hidden, n_output, dropout=(.2, .2), activation='ReLU'):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_output)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward implementation of 3-layer decoder """
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # final layer
        act = self.lin3(act)
        act = F.sigmoid(act)
        return act


TORCH_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}


class DenoisingAutoEncoder():
    def __init__(self,
                 n_hidden=100,
                 n_code=50,
                 lr=0.001,
                 batch_size=100,
                 n_epochs=500,
                 optimizer='adam',
                 normalize_inputs=True,
                 activation='ReLU',
                 dropout=(.2, .2),
                 noise_factor=0.2,
                 verbose=True):

        self.enc, self.dec = None, None
        self.n_hidden = n_hidden
        self.n_code = n_code
        self.n_epochs = n_epochs
        self.optimizer = optimizer.lower()
        self.normalize_inputs = normalize_inputs
        self.verbose = verbose
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.activation = activation
        self.noise_factor = noise_factor
        self.loss = nn.MSELoss()

    # TODO Corrupt condition too?
    def corrupt(self, batch, condition=None):
        """ Add noise to the input """
        noise = torch.randn(batch.size()) * self.noise_factor
        if torch.cuda.is_available():
            noise = noise.cuda()
        return (batch + noise)

    def eval(self):
        """ Put all NN modules into eval mode """
        self.enc.eval()
        self.dec.eval()

    def train(self):
        """ Put all NN modules into train mode """
        self.enc.train()
        self.dec.train()

    def zero_grad(self):
        """ Zeros gradients of all NN modules """
        self.enc.zero_grad()
        self.dec.zero_grad()

    def ae_step(self, batch, condition=None):
        """ Perform one autoencoder training step """
        z_sample = self.enc(self.corrupt(batch))
        if condition is not None:
            z_sample = torch.cat((z_sample, condition), 1)

        x_sample = self.dec(z_sample)
        recon_loss = self.loss(z_sample,batch)
        recon_loss.backward()
        self.enc_optim.step()
        self.dec_optim.step()
        self.zero_grad()
        return recon_loss.data[0].item()

    def partial_fit(self, X, y=None, condition=None):
        """ Performs reconstrction, discimination, generator training steps """
        if y is not None:
            raise ValueError("(Semi-)supervised usage not supported")
        # Transform to Torch (Cuda) Variable, shift batch to GPU
        X = Variable(torch.FloatTensor(X))
        if torch.cuda.is_available():
            X = X.cuda()

        if condition is not None:
            condition = condition.astype('float32')
            if sp.issparse(condition):
                condition = condition.toarray()
            condition = Variable(torch.from_numpy(condition))
            if torch.cuda.is_available():
                condition = condition.cuda()

        # Make sure we are in training mode and zero leftover gradients
        self.train()
        self.zero_grad()
        # One step each, could balance
        recon_loss = self.ae_step(X, condition=condition)
        if self.verbose:
            log_losses(recon_loss, 0, 0)
        return self

    def fit(self, X, y=None, condition=None):
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")

        self.enc = Encoder(X.shape[1], self.n_hidden, self.n_code,
                           final_activation='linear',
                           normalize_inputs=self.normalize_inputs,
                           dropout=self.dropout, activation=self.activation)
        if condition is not None:
            self.dec = Decoder(self.n_code + condition.shape[1], self.n_hidden,
                               X.shape[1], dropout=self.dropout, activation=self.activation)
            assert condition.shape[0] == X.shape[0]
        else:
            self.dec = Decoder(self.n_code, self.n_hidden, X.shape[1],
                               dropout=self.dropout, activation=self.activation)

        if torch.cuda.is_available():
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
        optimizer_gen = TORCH_OPTIMIZERS[self.optimizer]
        # Reconstruction
        self.enc_optim = optimizer_gen(self.enc.parameters(), lr=self.lr)
        self.dec_optim = optimizer_gen(self.dec.parameters(), lr=self.lr)

        # do the actual training
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("Epoch", epoch + 1)

            # Shuffle on each new epoch
            if condition is not None:
                X_shuf, condition_shuf = sklearn.utils.shuffle(X, condition)
            else:
                X_shuf = sklearn.utils.shuffle(X)

            for start in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuf[start:(start + self.batch_size)].toarray()
                # condition may be None
                if condition is not None:
                    c_batch = condition_shuf[start:(start + self.batch_size)]
                    self.partial_fit(X_batch, condition=c_batch)
                else:
                    self.partial_fit(X_batch)

            if self.verbose:
                # Clean up after flushing batch loss printings
                print()
        return self

    def predict(self, X, condition=None):
        self.eval()  # Deactivate dropout
        pred = []
        for start in range(0, X.shape[0], self.batch_size):
            # batched predictions, yet inclusive
            X_batch = X[start:(start + self.batch_size)].toarray()
            X_batch = torch.FloatTensor(X_batch)
            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
            X_batch = Variable(X_batch)

            if condition is not None:
                c_batch = condition[start:(start + self.batch_size)]
                if sp.issparse(c_batch):
                    c_batch = c_batch.toarray()
                c_batch = torch.FloatTensor(c_batch)
                if torch.cuda.is_available():
                    c_batch = c_batch.cuda()
                c_batch = Variable(c_batch)

            # reconstruct
            z = self.enc(X_batch)
            if condition is not None:
                z = torch.cat((z, c_batch), 1)
            X_reconstuction = self.dec(z)
            # shift
            X_reconstuction = X_reconstuction.data.cpu().numpy()
            pred.append(X_reconstuction)
        return np.vstack(pred)


class DAERecommender(Recommender):
    """
    Denoising Recommender
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

    def __init__(self, tfidf_params=dict(),
                 **kwargs):
        """ tfidf_params get piped to either TfidfVectorizer or
        EmbeddedVectorizer.  Remaining kwargs get passed to
        AdversarialAutoencoder """
        super().__init__()
        self.verbose = kwargs.get('verbose', True)
        self.use_title = kwargs.pop('use_title', False)
        self.embedding = kwargs.pop('embedding', None)
        self.vect = None
        self.dae_params = kwargs
        self.tfidf_params = tfidf_params

    def __str__(self):
        desc = "Denoising Autoencoder using titles: " + ("Yes!" if self.use_title else "No.")
        desc += '\nDAE Params: ' + str(self.dae_params)
        desc += '\nTfidf Params: ' + str(self.tfidf_params)
        return desc

    def train(self, training_set):
        X = training_set.tocsr()
        if self.use_title:
            if self.embedding:
                self.vect = GensimEmbeddedVectorizer(self.embedding,
                                                     **self.tfidf_params)
            else:
                self.vect = TfidfVectorizer(**self.tfidf_params)

            titles = training_set.get_attribute("title")
            titles = self.vect.fit_transform(titles)
            assert titles.shape[0] == X.shape[0], "Dims dont match"
            # X = sp.hstack([X, titles])
        else:
            titles = None

        self.dae = DenoisingAutoEncoder(**self.dae_params)

        self.dae.fit(X, condition=titles)

    def predict(self, test_set):
        X = test_set.tocsr()
        if self.use_title:
            # Use titles as condition
            titles = test_set.get_attribute("title")
            titles = self.vect.transform(titles)
            pred = self.dae.predict(X, condition=titles)
        else:
            pred = self.dae.predict(X)

        return pred


def main():
    """ Evaluates the DAE Recommender """
    CONFIG = {
        'pub': ('/data21/lgalke/datasets/citations_pmc.tsv', 2011, 50),
        'eco': ('/data21/lgalke/datasets/econbiz62k.tsv', 2012, 1)
    }

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('data', type=str, choices=['pub', 'eco'])
    args = PARSER.parse_args()
    DATA = CONFIG[args.data]
    logfile = '/data22/ivagliano/test-vae/' + args.data + '-decoder.log'
    bags = Bags.load_tabcomma_format(DATA[0])
    c_year = DATA[1]

    evaluate = Evaluation(bags,
                          year=c_year,
                          logfile=logfile).setup(min_count=DATA[2],
                                                 min_elements=2)
    # print("Loading pre-trained embedding", W2V_PATH)
    vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

    params = {
        'n_epochs': 1,
        'batch_size': 100,
        'optimizer': 'adam',
        'normalize_inputs': True,
        # 'prior': 'gauss',
    }
    # 100 hidden units, 200 epochs, bernoulli prior, normalized inputs -> 0.174
    activations = ['ReLU', 'SELU']
    lrs = [(0.001, 0.0005), (0.001, 0.001)]
    hcs = [(100, 50), (300, 100)]

    # dropouts = [(.2,.2), (.1,.1), (.1, .2), (.25, .25), (.3,.3)] # .2,.2 is best
    # priors = ['categorical'] # gauss is best
    # normal = [True, False]
    # bernoulli was good, letz see if categorical is better... No
    import itertools
    models = [DAERecommender(**params,
                             use_title=ut, embedding=vectors)
              for ut in itertools.product((True, False))]
    # models = [DecodingRecommender(embedding=vectors)]
    evaluate(models)


if __name__ == '__main__':
    main()
