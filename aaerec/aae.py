""" Adversarially Regualized Autoencoders """
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
from .ub import AutoEncoderMixin

# numpy
import numpy as np
import scipy.sparse as sp

# own recommender stuff
from .base import Recommender
from .datasets import Bags
from .evaluation import Evaluation
from .ub import GensimEmbeddedVectorizer
from gensim.models.keyedvectors import KeyedVectors

from .condition import ConditionList, _check_conditions


torch.manual_seed(42)
TINY = 1e-12

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True

STATUS_FORMAT = "[ R: {:.4f} | D: {:.4f} | G: {:.4f} ]"




def assert_condition_callabilities(conditions):
    raise DeprecationWarning("Use _check_conditions(conditions, condition_data) instead")
    if type(conditions) == type(True):
        pass
    else:
        assert type(conditions) != type("") and hasattr(conditions,'__iter__'), "Conditions needs to be a list of different conditions. It is a {} now.".format(type(conditions))

# TODO: pull this out, so its generally available
# TODO: put it into use at other points in class
# TODO: ensure features are appended correctly
def concat_side_info(vectorizer,training_set,side_info_subset):
    """
    Constructing an np.array with having the concatenated features in shape[1]
    :param training_set: Bag class dataset,
    :side_info_subset: list of str, the attribute keys in Bag class
    :return:
    """
    raise DeprecationWarning("Use ConditionList.encode_impose(...) instead")
    attr_vect = []
    # ugly substitute for do_until pattern
    for i, attribute in enumerate(side_info_subset):
        attr_data = training_set.get_single_attribute(attribute)
        if i < 1:
            attr_vect = vectorizer.fit_transform(attr_data)
        else:
            # rows are instances, cols are features --> adding cols makes up new features
            attr_vect = np.concatenate((attr_vect, vectorizer.fit_transform(attr_data)), axis=1)
    return attr_vect

def log_losses(*losses):
    print('\r'+STATUS_FORMAT.format(*losses), end='', flush=True)

def sample_categorical(size):
    batch_size, n_classes = size
    cat = np.random.randint(0, n_classes, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat

def sample_bernoulli(size):
    ber = np.random.randint(0, 1, size).astype('float32')
    return torch.from_numpy(ber)


PRIOR_SAMPLERS = {
    'categorical': sample_categorical,
    'bernoulli': sample_bernoulli,
    'gauss': torch.randn
}

PRIOR_ACTIVATIONS = {
    'categorical': 'softmax',
    'bernoulli': 'sigmoid',
    'gauss': 'linear'
}




class Encoder(nn.Module):
    """ Three-layer Encoder """
    def __init__(self, n_input, n_hidden, n_code, final_activation=None,
                 normalize_inputs=True, dropout=(.2,.2), activation='ReLU'):
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
    def __init__(self, n_code, n_hidden, n_output, dropout=(.2,.2), activation='ReLU'):
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


class Discriminator(nn.Module):
    """ Discriminator """
    def __init__(self, n_code, n_hidden, dropout=(.2,.2), activation='ReLU'):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, 1)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward of 3-layer discriminator """
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)

        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)



        # act = F.dropout(self.lin1(inp), p=self.dropout[0], training=self.training)
        # act = F.relu(act)
        # act = F.dropout(self.lin2(act), p=self.dropout[1], training=self.training)
        # act = F.relu(act)
        return F.sigmoid(self.lin3(act))


TORCH_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}

class AutoEncoder():
    ### DONE Adapt to generic condition ###
    def __init__(self,
                 n_hidden=100,
                 n_code=50,
                 lr=0.001,
                 batch_size=100,
                 n_epochs=500,
                 optimizer='adam',
                 normalize_inputs=True,
                 activation='ReLU',
                 dropout=(.2,.2),
                 conditions=None,
                 verbose=True):

        ### TODO Adapt to generic condition ###
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

        self.conditions = conditions

    def eval(self):
        """ Put all NN modules into eval mode """
        ### DONE Adapt to generic condition ###
        self.enc.eval()
        self.dec.eval()
        self.conditions.eval()

    def train(self):
        """ Put all NN modules into train mode """
        ### DONE Adapt to generic condition ###
        self.enc.train()
        self.dec.train()
        self.conditions.train()


    def ae_step(self, batch, condition_data=None):
        """
        Perform one autoencoder training step
            :param batch: np.array, the base data from Bag class
            :param condition: condition_matrix: np.array, feature space of side_info
            :return: binary_cross_entropy for this step
            """
        ### DONE Adapt to generic condition ###

        # why is this double to AdversarialAutoEncoder? Lukas: it's likely the two models will diverge
        # what is relationship to train in DecodingRecommender? Train only uses Condition. Those are implementet seperately
        # assert_condition_callabilities(condition_matrix)
        z_sample = self.enc(batch)

        # condition_matrix is already a matrix and doesn't need to be concatenated again
        # TODO: think/ask: where is it better to do concat? Here or when first  setted up for training
        # IMO: when setting up for training, because it's the used downstream all the same

        # concat base data with side_info
        # z_sample = torch.cat((z_sample, condition_matrix), 1)

        use_condition = _check_conditions(self.conditions, condition_data)
        if use_condition:
            z_sample = self.conditions.encode_impose(z_sample, condition_data)

        x_sample = self.dec(z_sample)
        recon_loss = F.binary_cross_entropy(x_sample + TINY,
                                            batch.view(batch.size(0),
                                                       batch.size(1)) + TINY)
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        self.conditions.zero_grad()
        recon_loss.backward()
        self.enc_optim.step()
        self.dec_optim.step()
        self.conditions.step()
        return recon_loss.item()

    def partial_fit(self, X, y=None, condition_data=None):
        """
            Performs reconstrction, discimination, generator training steps
        :param X: np.array, the base data from Bag class
        :param y: dummy variable, throws Error if used
        :param condition_matrix: np.array, feature space of side_info
        :return:
        """
        ### DONE Adapt to generic condition ###
        _check_conditions(self.conditions, condition_data)

        if y is not None:
            raise ValueError("(Semi-)supervised usage not supported")
        # Transform to Torch (Cuda) Variable, shift batch to GPU
        X = Variable(torch.FloatTensor(X))
        if torch.cuda.is_available():
            X = X.cuda()

        # if condition_matrix is not None:
        #     condition_matrix = condition_matrix.astype('float32')
        #     if sp.issparse(condition_matrix):
        #         condition_matrix = condition_matrix.toarray()
        #     condition_matrix = Variable(torch.from_numpy(condition_matrix))
        #     if torch.cuda.is_available():
        #         condition_matrix = condition_matrix.cuda()

        # Make sure we are in training mode and zero leftover gradients
        self.train()
        # One step each, could balance
        recon_loss = self.ae_step(X, condition_data=condition_data)
        if self.verbose:
            log_losses(recon_loss, 0, 0)
        return self

    def fit(self, X, y=None, condition_data=None):
        """
        :param X: np.array, the base data from Bag class
        :param y: dummy variable, throws Error if used
        :param condition_matrix: np.array, feature space of side_info
        :return:
        """
        ### DONE Adapt to generic condition ###
        # TODO: check how X representation and numpy.array work together
        # TODO: adapt combining X and new_conditions_name
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")

        use_condition = _check_conditions(self.conditions, condition_data)

        if use_condition:
            code_size = self.n_code + self.conditions.size_increment()
        else:
            code_size = self.n_code


        # Encoder just gets BaseData (~X), Decoder just Encoding and SideInfo
        # but the the dims mismatch
        # TODO: Fix/Find Encoding step in predict
        print("encoder dims", X.shape[1])
        self.enc = Encoder(X.shape[1], self.n_hidden, self.n_code,
                           final_activation='linear',
                           normalize_inputs=self.normalize_inputs,
                           dropout=self.dropout, activation=self.activation)
        # if condition_matrix is not None:
        #     # seems to be not enough TODO: check what is done in decoder so that dims fit
        #     # TODO: find out why dims are arbitrary
        #     # [100 x 381], m2: [1616 x 100] vs [100 x 376], m2: [1628 x 100]
        #     assert condition_matrix.shape[0] == X.shape[0]
        #     print("condition_matrix shape: ",condition_matrix.shape,"X.shape", X.shape)
        #     # (3600, 1567) (3600, 88323), (3600, 1566) (3600, 87305),  (3600, 1575) (3600, 86911)
        #     # data set is stable: total: 4000 records with 269755 ratings
        #     # on master branch there are values in all [ R: 0.6524 | D: 1.3585 | G: 0.7273 ]
        #     # shape[1] is the length of feature space --> this prob gives how many dims for Decoder
        self.dec = Decoder(code_size, self.n_hidden,
                           X.shape[1], dropout=self.dropout, activation=self.activation)

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
        :param condition_matrix: np.array, feature space of side_info
        :return:
        """
        ### DONE Adapt to generic condition ###
        # TODO: first look into fit, as predict is based on that!!!
        use_condition = _check_conditions(self.conditions, condition_data)
        self.eval()  # Deactivate dropout
        self.conditions.eval()
        pred = []
        for start in range(0, X.shape[0], self.batch_size):
            # batched predictions, yet inclusive
            end = start + self.batch_size
            X_batch = X[start:end].toarray()
            X_batch = torch.FloatTensor(X_batch)
            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
            X_batch = Variable(X_batch)

            if use_condition:
                c_batch = [c[start:end] for c in condition_data_shuf]

            z = self.enc(X_batch)
            if use_condition:
                z = self.conditions.encode_impose(z, condition_data=c_batch)
            # reconstruct
            # Encoder is set in fit() method
            # TODO: find why it throws. seems to be dims mismatch
            # File "/home/gerstenkorn/anaconda3/envs/citation/lib/python3.6/site-packages/torch/nn/functional.py", line 1024, in linear
            # return torch.addmm(bias, input, weight.t())
            #            RuntimeError: size mismatch, m1: [100 x 376], m2: [1628 x 100] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:2070
            # other iteration:  RuntimeError: size mismatch, m1: [100 x 387], m2: [1627 x 100] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:940

            X_reconstuction = self.dec(z)
            # shift
            X_reconstuction = X_reconstuction.data.cpu().numpy()
            pred.append(X_reconstuction)
        return np.vstack(pred)


class DecodingRecommender(Recommender):
    """ Only the decoder part of the AAE, basically 2-MLP """
    ### TODO Adapt to generic condition ###
    def __init__(self, n_epochs=100, batch_size=100, optimizer='adam',
                 n_hidden=100, embedding=None,
                 lr=0.001, verbose=True, tfidf_params={},
                 **mlp_params):
        ### TODO Adapt to generic condition ###
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer.lower()
        self.mlp_params = mlp_params
        self.verbose = verbose
        self.embedding = embedding
        self.tfidf_params = tfidf_params
        self.n_hidden = n_hidden

        self.mlp, self.mlp_optim, self.vect = None, None, None

    def __str__(self):
        ### TODO Adapt to generic condition ###
        desc = "MLP-2 Decoder with " + str(self.n_hidden) + " hidden units"
        desc += " training for " + str(self.n_epochs)
        desc += " optimized by " + self.optimizer
        desc += " with learning rate " + str(self.lr)
        desc += "\nUsing embedding: " +  ("Yes" if self.embedding is not None else "No")
        desc += "\n MLP Params: " + str(self.mlp_params)
        desc += "\n Tfidf Params: " + str(self.tfidf_params)
        return desc

    def partial_fit(self, X, y):
        ### TODO Adapt to generic condition ###
        self.mlp.train()
        self.mlp.zero_grad()
        if sp.issparse(X):
            X = X.toarray()
        if sp.issparse(y):
            y = y.toarray()
        X = Variable(torch.FloatTensor(X))
        y = Variable(torch.FloatTensor(y))
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
        y_pred = self.mlp(X)
        loss = F.binary_cross_entropy(y_pred + TINY, y + TINY)
        loss.backward()
        self.mlp_optim.step()
        if self.verbose:
            print("\rLoss: {}".format(loss.data.item()), flush=True, end='')
        return self

    def fit(self, X, y):
        ### TODO Adapt to generic condition ###
        self.mlp = Decoder(X.shape[1], self.n_hidden, y.shape[1], **self.mlp_params)
        if torch.cuda.is_available():
            self.mlp = self.mlp.cuda()
        optimizer_gen = TORCH_OPTIMIZERS[self.optimizer]
        self.mlp_optim = optimizer_gen(self.mlp.parameters(), lr=self.lr)
        for __epoch in range(self.n_epochs):
            X_shuf, y_shuf = sklearn.utils.shuffle(X, y)
            for start in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuf[start:(start+self.batch_size)]
                if sp.issparse(X_batch):
                    X_batch = X_batch.toarray()
                y_batch = y_shuf[start:(start+self.batch_size)].toarray()
                if sp.issparse(y_batch):
                    y_batch = y_batch.toarray()
                self.partial_fit(X_batch, y_batch)

            if self.verbose:
                print()

        return self

    def train(self, training_set):
        ### TODO Adapt to generic condition ###
        # Fit function from condition to X
        X = training_set.tocsr()
        if self.embedding:
            self.vect = GensimEmbeddedVectorizer(self.embedding, **self.tfidf_params)
        else:
            self.vect = TfidfVectorizer(**self.tfidf_params)

        # TODO: add other side_infos separately
        # TODO: propagate setting condition upwards to calling methods
        # TODO: overthink where to do this (should belong to preprocessing, not in model)
        # TODO: Do it here to test, will be integrated in preprocessing with condition class

        condition = training_set.get_single_attribute("title")
        # this is specific to the title (and other textual features)
        # TODO: potentially adapt other vectorizer for non-textual features
        condition = self.vect.fit_transform(condition)
        print("{} distinct words in condition" .format(len(self.vect.vocabulary_)))
        self.fit(condition, X)


    def predict(self, test_set):
        ### TODO Adapt to generic condition ###
        # condition = test_set.get_single_attribute("title")
        # condition = self.vect.transform(condition).toarray()
        # condition = torch.FloatTensor(condition)
        # if torch.cuda.is_available():
        #     condition = condition.cuda()
        # self.mlp.eval()
        # x_pred = self.mlp(Variable(condition))
        # Batched variant to save gpu memory
        condition = test_set.get_single_attribute("title")
        condition = self.vect.transform(condition)
        self.mlp.eval()
        batch_results = []
        for start in range(0, condition.shape[0], self.batch_size):
            batch = condition[start:(start+self.batch_size)]
            if sp.issparse(batch):
                batch = batch.toarray()
            batch = torch.FloatTensor(batch)
            # Shift data to gpu
            if torch.cuda.is_available():
                batch = batch.cuda()
            res = self.mlp(Variable(batch, requires_grad=False))
            # Shift results back to cpu
            batch_results.append(res.cpu().detach().numpy())
        
        x_pred = np.vstack(batch_results)
        assert x_pred.shape[0] == condition.shape[0]
        return x_pred




class AdversarialAutoEncoder(AutoEncoderMixin):
    """ Adversarial Autoencoder """
    ### DONE Adapt to generic condition ###
    def __init__(self,
                 n_hidden=100,
                 n_code=50,
                 gen_lr=0.001,
                 reg_lr=0.001,
                 prior='gauss',
                 prior_scale=None,
                 batch_size=100,
                 n_epochs=500,
                 optimizer='adam',
                 normalize_inputs=True,
                 activation='ReLU',
                 dropout=(.2, .2),
                 conditions=None,
                 verbose=True):
        # Build models
        self.prior = prior.lower()
        self.prior_scale = prior_scale

        # Encoder final activation depends on prior distribution
        self.prior_sampler = PRIOR_SAMPLERS[self.prior]
        self.encoder_activation = PRIOR_ACTIVATIONS[self.prior]
        self.optimizer = optimizer.lower()

        self.n_hidden = n_hidden
        self.n_code = n_code
        self.gen_lr = gen_lr
        self.reg_lr = reg_lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.gen_lr, self.reg_lr = gen_lr, reg_lr
        self.n_epochs = n_epochs

        self.enc, self.dec, self.disc = None, None, None
        self.enc_optim, self.dec_optim = None, None
        self.gen_optim, self.disc_optim = None, None
        self.normalize_inputs = normalize_inputs

        self.dropout = dropout
        self.activation = activation

        self.conditions = conditions

    def __str__(self):
        desc = "Adversarial Autoencoder"
        n_h, n_c = self.n_hidden, self.n_code
        gen, reg = self.gen_lr, self.reg_lr
        desc += " ({}, {}, {}, {}, {})".format(n_h, n_h, n_c, n_h, n_h)
        desc += " optimized by " + self.optimizer
        desc += " with learning rates Gen, Reg = {}, {}".format(gen, reg)
        desc += ", using a batch size of {}".format(self.batch_size)
        desc += "\nMatching the {} distribution".format(self.prior)
        desc += " by {} activation.".format(self.encoder_activation)
        return desc

    def eval(self):
        """ Put all NN modules into eval mode """
        ### DONE Adapt to generic condition ###
        self.enc.eval()
        self.dec.eval()
        self.disc.eval()
        if self.conditions:
            # Forward call to condition modules
            self.conditions.eval()

    def train(self):
        """ Put all NN modules into train mode """
        ### DONE Adapt to generic condition ###
        self.enc.train()
        self.dec.train()
        self.disc.train()
        if self.conditions:
            # Forward call to condition modules
            self.conditions.train()

    def zero_grad(self):
        """ Zeros gradients of all NN modules """
        self.enc.zero_grad()
        self.dec.zero_grad()
        self.disc.zero_grad()

    # why is this double? to AdversarialAutoEncoder
    def ae_step(self, batch, condition_data=None):
        ### DONE Adapt to generic condition ###
        """
        # why is this double? to AdversarialAutoEncoder => THe AE Step is very different from plain AEs
        # what is relationship to train?
        # Condition is used explicitly here, and hard coded but non-explicitly here
        Perform one autoencoder training step
        :param batch:
        :param condition: ??? ~ training_set.get_single_attribute("title") <~ side_info = unpack_playlists(playlists)
        :return:
        """
        print("batch",batch,"condition", conditions_batch)
        z_sample = self.enc(batch)
        if condition_data:
            self.conditions.encode_impose(z_sample, conditions_batch)

        x_sample = self.dec(z_sample)
        recon_loss = F.binary_cross_entropy(x_sample + TINY,
                                            batch.view(batch.size(0),
                                                       batch.size(1)) + TINY)
        # Clear all related gradients
        self.enc.zero_grad()
        self.dec.zero_grad()
        self.conditions.zero_grad()
    
        # Compute gradients
        recon_loss.backward()

        # Update parameters
        self.enc_optim.step()
        self.dec_optim.step()
        self.conditions.step()
        return recon_loss.data[0].item()

    def disc_step(self, batch):
        """ Perform one discriminator step on batch """
        self.enc.eval()
        z_real = Variable(self.prior_sampler((batch.size(0), self.n_code)))
        if self.prior_scale is not None:
            z_real = z_real * self.prior_scale

        if torch.cuda.is_available():
            z_real = z_real.cuda()
        z_fake = self.enc(batch)

        # Compute discrimnator outputs and loss
        disc_real_out, disc_fake_out = self.disc(z_real), self.disc(z_fake)
        disc_loss = -torch.mean(torch.log(disc_real_out + TINY)
                                + torch.log(1 - disc_fake_out + TINY))
        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()
        return disc_loss.data[0].item()

    def gen_step(self, batch):
        self.enc.train()
        z_fake_dist = self.enc(batch)
        disc_fake_out = self.disc(z_fake_dist)
        gen_loss = -torch.mean(torch.log(disc_fake_out + TINY))
        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()
        return gen_loss.data[0].item()

    def partial_fit(self, X, y=None, condition_data=None):
        ### DONE Adapt to generic condition ###
        """ Performs reconstrction, discimination, generator training steps """
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")
        # Transform to Torch (Cuda) Variable, shift batch to GPU
        X = Variable(torch.FloatTensor(X))
        if torch.cuda.is_available():
            # Put batch on CUDA device!
            X = X.cuda()
       
        # if condition is not None:
        #     condition = condition.astype('float32')
        #     if sp.issparse(condition):
        #         condition = condition.toarray()
        #     condition = Variable(torch.from_numpy(condition))
        #     if torch.cuda.is_available():
        #         condition = condition.cuda()

        # Make sure we are in training mode and zero leftover gradients
        self.train()
        # One step each, could balance
        recon_loss = self.ae_step(X, condition_data=condition_data)
        disc_loss = self.disc_step(X)
        gen_loss = self.gen_step(X)
        if self.verbose:
            log_losses(recon_loss, disc_loss, gen_loss)
        return self

    def fit(self, X, y=None, condition_data=None):
        ### DONE Adapt to generic condition ###
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")

        use_condition = _check_conditions(self.conditions, condition_data)

        if use_condition:
            code_size = self.n_code + self.conditions.size_increment()
        else:
            code_size = self.n_code

        self.enc = Encoder(X.shape[1], self.n_hidden, self.n_code,
                           final_activation=self.encoder_activation,
                           normalize_inputs=self.normalize_inputs,
                           activation=self.activation,
                           dropout=self.dropout)
        self.dec = Decoder(code_size, self.n_hidden, X.shape[1],
                           activation=self.activation, dropout=self.dropout)

        self.disc = Discriminator(self.n_code, self.n_hidden,
                                  dropout=self.dropout,
                                  activation=self.activation)

        if torch.cuda.is_available():
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.disc = self.disc.cuda()
        optimizer_gen = TORCH_OPTIMIZERS[self.optimizer]
        # Reconstruction
        self.enc_optim = optimizer_gen(self.enc.parameters(), lr=self.gen_lr)
        self.dec_optim = optimizer_gen(self.dec.parameters(), lr=self.gen_lr)
        # Regularization
        self.gen_optim = optimizer_gen(self.enc.parameters(), lr=self.reg_lr)
        self.disc_optim = optimizer_gen(self.disc.parameters(), lr=self.reg_lr)

        # do the actual training
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("Epoch", epoch + 1)

            # Shuffle on each new epoch
            if use_condition:
                # shuffle(*arrays) takes several arrays and shuffles them so indices are still matching
                X_shuf, *condition_data_shuf = sklearn.utils.shuffle(X, *condition_data)
            else:
                X_shuf = sklearn.utils.shuffle(X)

            for start in range(0, X_shuf.shape[0], self.batch_size):
                end = start + self.batch_size

                # Make the batch dense!
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

    # def transform(self, X):
    #     X = Variable(torch.FloatTensor(X))
    #     if torch.cuda.is_available():
    #         X = X.cuda()
    #     return self.enc(X).data.cpu().numpy()

    # def inverse_transform(self, X):
    #     # doesnt work with numpy
    #     return self.dec(X)

    def predict(self, X, condition_data=None):
        ### DONE Adapt to generic condition ###
        self.eval()  # Deactivate dropout
        # In case some of the conditions has dropout
        self.conditions.eval()
        pred = []
        for start in range(0, X.shape[0], self.batch_size):
            # batched predictions, yet inclusive
            X_batch = X[start:(start+self.batch_size)]
            if sp.issparse(X_batch):
                X_batch = X_batch.toarray()
            X_batch = Variable(torch.FloatTensor(X_batch))
            if torch.cuda.is_available():
                X_batch = X_batch.cuda()

            # if condition_data is not None:
            #     c_batch = condition[start:(start+self.batch_size)]
            #     c_batch = c_batch.astype('float32')
            #     if sp.issparse(c_batch):
            #         c_batch = c_batch.toarray()
            #     c_batch = Variable(torch.from_numpy(c_batch))
            #     if torch.cuda.is_available():
            #         c_batch = c_batch.cuda()

            # reconstruct
            z = self.enc(X_batch)
            if condition_data is not None:
                # z = torch.cat((z, c_batch), 1)
                z = self.conditions.encode_impose(y, condition_data)
            X_reconstuction = self.dec(z)
            # shift
            X_reconstuction = X_reconstuction.data.cpu().numpy()
            pred.append(X_reconstuction)
        return np.vstack(pred)




class AAERecommender(Recommender):
    ### DONE Adapt to generic condition ###
    """
    Adversarially Regularized Recommender
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
    def __init__(self, adversarial=True, conditions=None, **kwargs):
        ### DONE Adapt to generic condition ###
        """ tfidf_params get piped to either TfidfVectorizer or
        EmbeddedVectorizer.  Remaining kwargs get passed to
        AdversarialAutoencoder """
        super().__init__()
        self.verbose = kwargs.get('verbose', True)

        # self.use_side_info = kwargs.pop('use_side_info', False)

        self.conditions = conditions

        # assert_condition_callabilities(self.use_side_info)
        # Embedding is now part of a condition
        # self.embedding = kwargs.pop('embedding', None)
        # Vectorizer also...
        # self.vect = None
        self.model_params = kwargs
        # tfidf params now need to be in the respective *condition* of condition_list
        # self.tfidf_params = tfidf_params
        self.adversarial = adversarial

    def __str__(self):
        ### DONE Adapt to generic condition ###
        if self.adversarial:
            desc = "Adversarial Autoencoder"
        else:
            desc = "Autoencoder"

        if self.conditions:
            desc += " conditioned on " ','.join(self.conditions.keys())
        desc += '\Model Params: ' + str(self.model_params)
        # TODO: is it correct for self.tfidf_params to be an EMPTY dict
        # DONE: Yes it is only the *default*!
        # desc += '\nTfidf Params: ' + str(self.tfidf_params)
        # Anyways, this kind of stuff goes into the condition itself
        return desc


    def train(self, training_set):
        ### DONE Adapt to generic condition ###
        """
        1. get basic representation
        2. ? add potential side_info in ??? representation
        3. initialize a (Adversarial) Autoencoder variant
        4. fit based on Autoencoder
        :param training_set: ???, Bag Class training set
        :return: trained self
        """
        X = training_set.tocsr()
        if self.conditions:
            condition_data_raw = training_set.get_attributes(self.conditions.keys())
            condition_data = self.conditions.fit_transform(condition_data_raw)
        else:
            condition_data = None

        # X seems to be a "special" case formatting for input. TODO: check representation in function call tocsr()
        # if self.use_side_info:



        #     # TODO: later with condition: use attribute respective vectorizer
        #     if self.embedding:
        #         self.vect = GensimEmbeddedVectorizer(self.embedding,
        #                                              **self.tfidf_params)
        #     else:
        #         self.vect = TfidfVectorizer(**self.tfidf_params)




        #     attr_vect = concat_side_info(self.vect,training_set,side_info_subset=self.use_side_info)
        #     assert attr_vect.shape[0] == X.shape[0], "Dims dont match"




            # möglichkeit zum anderen vectorisieren -> muss für kleine Batches dann acuh gemacht werden --> in klasse speichern
            # Lukas Idee: Objektorientiert ~"condition" beerbt von nn.module  .encode um batch von callback (= mitgegebene funktion) transformieren
            # in dem Falll in forward methode
            # wie kann man das anhand vom globalen Loss abhängig ändern, da on the fly gelernt
            # und nicht im preprocessing. ~~> etwas wie backpropagation
            # auf loss .backward aufrufbar --> backprobagation entsprechend berechnet
            # nn.module wird viel beerbt. ~get_attributes bekommt man alle names aus modul
            #

        # else:
        #     attr_vect = None

        if self.adversarial:
            # Pass conditions through along with hyperparams
            self.model = AdversarialAutoEncoder(conditions=self.conditions, **self.model_params)
        else:
            # Pass conditions through along with hyperparams!
            self.model = AutoEncoder(conditions=self.conditions, **self.model_params)

        # gives (Adversarial) Autoencoder BaseData (--> X: <???> representation) and side_info (attr_vect: numpy)
        self.model.fit(X, condition_data=condition_data)

    def predict(self, test_set):
        ### DONE Adapt to generic condition ###
        X = test_set.tocsr()
        if self.conditions:
            condition_data_raw = test_set.get_attributes(self.conditions.keys())
            # Important to not call fit here, but just transform
            condition_data = self.conditions.transform(condition_data_raw)
        else:
            condition_data = None

        # if self.use_side_info:
        #     # change the attributes/conditions/side_infos here

        #     attr_vect = concat_side_info(self.vect, test_set,side_info_subset=self.use_side_info)
        #     assert attr_vect.shape[0] == X.shape[0], "Dims dont match"

            # pred = self.aae.predict(X, condition_matrix=attr_vect)
        # else:
        pred = self.model.predict(X, condition_data=condition_data)

        return pred


def main():
    """ Evaluates the AAE Recommender """
    CONFIG = {
        'pub': ('../Data/PMC/citations_pmc.tsv', 2011, 50),
        'eco': ('../Data/Economics/econbiz62k.tsv', 2012, 1)
    }

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('data', type=str, choices=['pub','eco'])
    args = PARSER.parse_args()
    DATA = CONFIG[args.data]
    logfile = 'results/' + args.data + '-decoder.log'
    bags = Bags.load_tabcomma_format(DATA[0])
    c_year = DATA[1]


    evaluate = Evaluation(bags,
                          year=c_year,
                          logfile=logfile).setup(min_count=DATA[2],
                                                 min_elements=2)
    # print("Loading pre-trained embedding", W2V_PATH)
    vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

    params = {
        'n_epochs': 100,
        'batch_size': 100,
        'optimizer': 'adam',
        'normalize_inputs': True,
        'prior': 'gauss',
    }
    # 100 hidden units, 200 epochs, bernoulli prior, normalized inputs -> 0.174
    activations = ['ReLU','SELU']
    lrs = [(0.001, 0.0005), (0.001, 0.001)]
    hcs = [(100, 50), (300, 100)]

    # dropouts = [(.2,.2), (.1,.1), (.1, .2), (.25, .25), (.3,.3)] # .2,.2 is best
    # priors = ['categorical'] # gauss is best
    # normal = [True, False]
    # bernoulli was good, letz see if categorical is better... No
    import itertools
    models = [AAERecommender(**params, n_hidden=hc[0], n_code=hc[1],
                             use_title=ut, embedding=vectors,
                             gen_lr=lr[0], reg_lr=lr[1], activation=a)
              for ut, lr, hc, a in itertools.product((True, False), lrs, hcs, activations)]
    # models = [DecodingRecommender(embedding=vectors)]
    evaluate(models)


if __name__ == '__main__':
    main()
