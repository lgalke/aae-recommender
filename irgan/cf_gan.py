import tensorflow as tf
from irgan.dis_model import DIS
from irgan.gen_model import GEN
# import cPickle
import numpy as np
import irgan.utils as ut
import multiprocessing
import argparse

# own recommender stuff
from aaerec.base import Recommender
from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.ub import GensimEmbeddedVectorizer
from gensim.models.keyedvectors import KeyedVectors

cores = multiprocessing.cpu_count()

# workdir = 'ml-100k/'
DIS_TRAIN_FILE = 'dis-train.txt'

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

        self.generator = GEN(item_num, user_num, emb_dim, lamda=0.0 / batch_size, param=gen_param, initdelta=init_delta,
                             learning_rate=lr)
        self.discriminator = DIS(item_num, user_num, emb_dim, lamda=0.1 / batch_size, param=None, initdelta=init_delta,
                                 learning_rate=lr)

    def dcg_at_k(self, r, k):
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    def ndcg_at_k(self, r, k):
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k) / dcg_max

    # def simple_test_one_user(self, x):
    def simple_test_one_user(self, rating, u):
        # rating = x[0]
        # u = x[1]

        test_items = list(self.all_items - set(self.user_pos_train[u]))
        item_score = []
        for i in test_items:
            item_score.append((i, rating[i]))

        item_score = sorted(item_score, key=lambda x: x[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]

        r = []
        for i in item_sort:
            if i in self.user_pos_test[u]:
                r.append(1)
            else:
                r.append(0)

        p_3 = np.mean(r[:3])
        p_5 = np.mean(r[:5])
        p_10 = np.mean(r[:10])
        ndcg_3 = self.ndcg_at_k(r, 3)
        ndcg_5 = self.ndcg_at_k(r, 5)
        ndcg_10 = self.ndcg_at_k(r, 10)

        return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

    def generate_for_d(self, sess, model, filename):
        data = []
        for u in self.user_pos_train:
            pos = self.user_pos_train[u]

            rating = sess.run(model.all_rating, {model.u: [u]})
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

        # TODO use condition
        # use_condition = _check_conditions(self.conditions, condition_data)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        # print("gen ", self.predict(sess, self.generator))
        # print("dis ", self.predict(sess, self.discriminator))

        # TODO use proper logging file
        # dis_log = open(workdir + 'dis_log.txt', 'w')
        # gen_log = open(workdir + 'gen_log.txt', 'w')

        self.user_pos_train = X

        # minimax training
        # best = 0.
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("Epoch", epoch + 1)

            if epoch >= 0:
                for d_epoch in range(self.d_epochs): #100
                    if d_epoch % 5 == 0:
                        self.generate_for_d(self.sess, self.generator, DIS_TRAIN_FILE)
                        train_size = ut.file_len(DIS_TRAIN_FILE)
                    index = 1
                    while True:
                        if index > train_size:
                            break
                        if index + self.batch_size <= train_size + 1:
                            input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                    self.batch_size)
                        else:
                            input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                    train_size - index + 1)
                        index += self.batch_size

                        _ = self.sess.run(self.discriminator.d_updates,
                                     feed_dict={self.discriminator.u: input_user, self.discriminator.i: input_item,
                                                self.discriminator.label: input_label})

                # Train G
                for g_epoch in range(self.g_epochs):  # 50
                    for u in self.user_pos_train:
                        sample_lambda = 0.2
                        pos = self.user_pos_train[u]

                        rating = self.sess.run(self.generator.all_logits, {self.generator.u: u})
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
                        reward = self.sess.run(self.discriminator.reward, {self.discriminator.u: u,
                                                                           self.discriminator.i: sample})
                        reward = reward * prob[sample] / pn[sample]
                        ###########################################################################
                        # Update G
                        ###########################################################################
                        _ = self.sess.run(self.generator.gan_updates,
                                     {self.generator.u: u, self.generator.i: sample, self.generator.reward: reward})

                    # result = self.predict(sess, self.generator)
                    # print("epoch ", epoch, "gen: ", result)
                    # buf = '\t'.join([str(x) for x in result])
                    # gen_log.write(str(epoch) + '\t' + buf + '\n')
                    # gen_log.flush()
                    #
                    # p_5 = result[1]
                    # if p_5 > best:
                    #     print('best: ', result)
                    #     best = p_5
                    #     self.generator.save_model(sess, "ml-100k/gan_generator.pkl")

        # gen_log.close()
        # dis_log.close()

        return self

    def predict(self, X, condition_data=None):
        result = np.array([0.] * 6)
        # pool = multiprocessing.Pool(cores)
        batch_size = 128
        test_users = list(X.keys())
        test_user_num = len(test_users)
        index = 0
        while True:
            if index >= test_user_num:
                break
            user_batch = test_users[index:index + batch_size]
            index += batch_size

            user_batch_rating = self.sess.run(self.generator.all_rating, {self.generator.u: user_batch})
            # user_batch_rating_uid = list(zip(user_batch_rating, user_batch))
            # batch_result = pool.map(self.simple_test_one_user, user_batch_rating_uid)
            # result += self.simple_test_one_user(user_batch_rating_uid)
            result += self.simple_test_one_user(user_batch_rating, user_batch)
            # for re in batch_result:
            #     result += re

        # pool.close()
        ret = result / test_user_num
        ret = list(ret)
        return ret


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

    # print("load model...")
    # param = cPickle.load(open(workdir + "model_dns_ori.pkl"))

    CONFIG = {
        'pub': ('/data21/lgalke/datasets/citations_pmc.tsv', 2011, 50),
        'eco': ('/data21/lgalke/datasets/econbiz62k.tsv', 2012, 1)
    }

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
    # print("Loading pre-trained embedding", W2V_PATH)
    # vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
    user_num = evaluate.train_set.size()[0] + evaluate.test_set.size()[0]
    item_num = evaluate.train_set.size()[1]
    models = [IRGANRecommender(user_num, item_num,g_epochs=1,d_epochs=1,n_epochs=1)]
    evaluate(models)


if __name__ == '__main__':
    main()
