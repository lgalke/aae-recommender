from irgan.dis_model import Discriminator
from irgan.gen_model import Generator
import numpy as np
import irgan.utils as ut
import multiprocessing
import argparse

import torch

# own recommender stuff
from aaerec.base import Recommender
from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.ub import GensimEmbeddedVectorizer
from gensim.models.keyedvectors import KeyedVectors

from aaerec.condition import ConditionList, _check_conditions, PretrainedWordEmbeddingCondition

# workdir = 'ml-100k/'
DIS_TRAIN_FILE = 'dis-train.txt'

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True

class IRGAN():

    def __init__(self,
                 user_num,
                 item_num,
                 gen_param=None,
                 batch_size=16,
                 emb_dim=5,
                 lr=0.001,
                 init_delta=0.05,
                 g_epochs=50,
                 d_epochs=100,
                 n_epochs=15,
                 # TODO normalize input
                 # normalize_inputs=True,
                 conditions=None,
                 verbose=True):

        # self.normalize_inputs = normalize_inputs
        self.verbose = verbose
        self.batch_size = batch_size
        self.conditions = conditions
        self.emb_dim = emb_dim
        self.lr = lr
        self.init_delta = init_delta
        self.gen_param = gen_param
        self.n_epochs = n_epochs
        self.g_epochs = g_epochs
        self.d_epochs = d_epochs
        self.user_num = user_num
        self.item_num = item_num
        self.all_items = set(range(item_num))

        self.generator = Generator(item_num, user_num, emb_dim, lamda=0.0 / batch_size, param=gen_param,
                                   initdelta=init_delta, learning_rate=lr, conditions=conditions)
        self.discriminator = Discriminator(item_num, user_num, emb_dim, lamda=0.1 / batch_size, param=gen_param,
                                           initdelta=init_delta, learning_rate=lr, conditions=conditions)
        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

    def simple_test_one_user(self, x):
        rating = x[0]
        u = x[1]

        test_items = list(self.all_items - set(self.user_pos_train[u]))
        item_score = []
        for i in test_items:
            # pairs predicted item-score
            item_score.append((i, rating[i]))

        # sort predicted items by score
        item_score = sorted(item_score, key=lambda x: x[1])
        item_score.reverse()
        # generate a matrix row with score as probability
        pred = np.zeros(self.item_num)
        for item in item_score:
            pred[item[0]] = item[1]

        return pred

    def generate_for_d(self, filename, condition_data=None):
        data = []

        for u in self.user_pos_train:
            pos = self.user_pos_train[u]
            if self.conditions:
                rating = self.generator.all_rating(u, condition_data[int(u), :])
            else:
                rating = self.generator.all_rating(u)
            rating = rating.detach_().cpu().numpy()
            rating = np.array(rating[0]) / 0.2  # Temperature
            exp_rating = np.exp(rating)
            prob = exp_rating / np.sum(exp_rating)

            neg = np.random.choice(np.arange(self.item_num), size=len(pos), p=prob)
            for i in range(len(pos)):
                data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

        with open(filename, 'w')as fout:
            fout.write('\n'.join(data))

    def fit(self, X, y=None, condition_data=None):
        """
        :param X: np.array, the base data from Bag class
        :param y: dummy variable, throws Error if used
        :param condition_data: generic list of conditions
        :return:
        """

        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")

        use_condition = _check_conditions(self.conditions, condition_data)

        self.user_pos_train = X

        # minimax training
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("Epoch", epoch + 1)

            if epoch >= 0:
                for d_epoch in range(self.d_epochs): #100
                    if d_epoch % 5 == 0:
                        self.generate_for_d(DIS_TRAIN_FILE, condition_data)
                        train_size = ut.file_len(DIS_TRAIN_FILE)
                    index = 1
                    while True:
                        if index > train_size:
                            break
                        if index + self.batch_size <= train_size + 1:
                            input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                    self.batch_size)
                            end = index + self.batch_size
                        else:
                            input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                    train_size - index + 1)
                            end = train_size + 1

                        index += self.batch_size
                        if torch.cuda.is_available():
                            input_label = torch.tensor(input_label).cuda()
                        else:
                            input_label = torch.tensor(input_label)
                        if use_condition:
                            c_batch = [c[index:end] for c in condition_data]
                            D_loss = self.discriminator(input_user, input_item, input_label, c_batch)
                        else:
                            D_loss = self.discriminator(input_user, input_item, input_label)
                        self.discriminator.step(D_loss)

                    if self.verbose:
                        print("\r[D Epoch %d/%d] [loss: %f]" % (d_epoch, self.d_epochs, D_loss.item()))

                # Train G
                for g_epoch in range(self.g_epochs):  # 50
                    for u in self.user_pos_train:
                        sample_lambda = 0.2
                        pos = self.user_pos_train[u]

                        if use_condition:
                            rating = self.generator.all_logits(u, condition_data[int(u)])
                        else:
                            rating = self.generator.all_logits(u)
                        rating = rating.detach_().cpu().numpy()
                        exp_rating = np.exp(rating)
                        prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                        pn = (1 - sample_lambda) * prob
                        pn[pos] += sample_lambda * 1.0 / len(pos)
                        # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

                        # print('pn sum=',pn.sum())
                        # Normilize by probability sum to avoid np.random.choice error 'probability do not sum to one'
                        pn /= pn.sum()
                        sample = np.random.choice(np.arange(self.item_num), 2 * len(pos), p=pn)
                        ###########################################################################
                        # Get reward and adapt it with importance sampling
                        ###########################################################################
                        reward = self.discriminator.get_reward(u, sample)
                        reward = reward.detach_().cpu().numpy() * prob[sample] / pn[sample]
                        ###########################################################################
                        # Update G
                        ###########################################################################
                        if torch.cuda.is_available():
                            sample = torch.tensor(sample).cuda()
                            reward = torch.tensor(reward).cuda()
                        else:
                            sample = torch.tensor(sample)
                            reward = torch.tensor(reward)
                        if use_condition:
                            G_loss = self.generator(u, sample, reward, condition_data[int(u)])
                        else:
                            G_loss = self.generator(u, sample, reward)
                        self.generator.step(G_loss)

                    if self.verbose:
                        print("\r[G Epoch %d/%d] [loss: %f]" % (g_epoch, self.g_epochs, G_loss.item()))

        return self

    def predict(self, X, condition_data=None):
        batch_size = 128
        test_users = list(X.keys())
        test_user_num = len(test_users)
        index = 0
        pred = []
        while True:
            if index >= test_user_num:
                break
            user_batch = test_users[index:index + batch_size]
            index += batch_size

            user_batch_rating = self.generator.all_rating(user_batch, condition_data)
            user_batch_rating = user_batch_rating.detach_().cpu().numpy()
            # TODO encode_impose on user_batch_rating?
            for user_batch_rating_uid in zip(user_batch_rating, user_batch):
                pred.append(self.simple_test_one_user(user_batch_rating_uid))

        return pred


class IRGANRecommender(Recommender):
    ### DONE Adapt to generic condition ###
    """
    IRGAN Recommender
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
    def __init__(self, user_num, item_num, gen_param=None, conditions=None, **kwargs):
        """ tfidf_params get piped to either TfidfVectorizer or
        EmbeddedVectorizer.  Remaining kwargs get passed to
        AdversarialAutoencoder """
        super().__init__()
        self.verbose = kwargs.get('verbose', True)
        self.conditions = conditions
        self.model_params = kwargs
        self.gen_param = gen_param
        self.user_num = user_num
        self.item_num = item_num

    def __str__(self):
       desc = "IRGAN"

       if self.conditions:
          desc += " conditioned on: " + ', '.join(self.conditions.keys())
       desc += '\nModel Params: ' + str(self.model_params)
       return desc

    def train(self, training_set):
        ### DONE Adapt to generic condition ###
        """
        1. get basic representation
        2. ? add potential side_info in ??? representation
        3. initialize a IRGAN variant
        4. fit based on IRGAN
        :param training_set: ???, Bag Class training set
        :return: trained self
        """
        X = training_set.to_dict()
        if self.conditions:
            condition_data_raw = training_set.get_attributes(self.conditions.keys())
            condition_data = self.conditions.fit_transform(condition_data_raw)
        else:
            condition_data = None

        self.model = IRGAN(self.user_num, self.item_num, self.gen_param, conditions=self.conditions,
                           **self.model_params)

        print(self)
        print(self.model)
        print(self.conditions)

        self.model.fit(X, condition_data=condition_data)

    def predict(self, test_set):
        ### DONE Adapt to generic condition ###
        X = test_set.to_dict()
        if self.conditions:
            condition_data_raw = test_set.get_attributes(self.conditions.keys())
            # Important to not call fit here, but just transform
            condition_data = self.conditions.transform(condition_data_raw)
        else:
            condition_data = None

        pred = self.model.predict(X, condition_data=condition_data)

        return pred


def main():

    CONFIG = {
        'pub': ('/data21/lgalke/datasets/citations_pmc.tsv', 2011, 50),
        'eco': ('/data21/lgalke/datasets/econbiz62k.tsv', 2012, 1)
    }

    print("Loading pre-trained embedding", W2V_PATH)
    vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

    CONDITIONS = ConditionList([
        ('title', PretrainedWordEmbeddingCondition(vectors))
    ])

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('data', type=str, choices=['pub', 'eco'])
    args = PARSER.parse_args()
    DATA = CONFIG[args.data]
    logfile = '/data22/ivagliano/test-irgan/' + args.data + '-decoder.log'
    bags = Bags.load_tabcomma_format(DATA[0])
    c_year = DATA[1]

    evaluate = Evaluation(bags,
                          year=c_year,
                          logfile=logfile).setup(min_count=DATA[2],
                                                 min_elements=2)
    user_num = evaluate.train_set.size()[0] + evaluate.test_set.size()[0]
    item_num = evaluate.train_set.size()[1]
    models = [IRGANRecommender(user_num, item_num, g_epochs=1, d_epochs=1, n_epochs=1, conditions=CONDITIONS)]
    evaluate(models)


if __name__ == '__main__':
    main()
