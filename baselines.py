""" Baselines """
import numpy as np
from numpy.random import rand
from base import Recommender


class RandomBaseline(Recommender):
    """ Random Baseline """

    def __str__(self):
        return "RNDM baseline"

    def train(self, X):
        pass

    def predict(self, X):
        X = X.tocsr()
        random_predictions = rand(X.shape[0], X.shape[1])
        return random_predictions


class Countbased(Recommender):
    """ Item Co-Occurrence """
    def __init__(self, order=1):
        super().__init__()
        self.order = order

    def __str__(self):
        s = "Count-based Predictor"
        s += " (order {})".format(self.order)
        return s

    def train(self, X):
        X = X.tocsr()
        # Construct cooccurrence matrix
        self.cooccurences = X.T @ X
        for __ in range(0, self.order - 1):
            self.cooccurences = self.cooccurences.T @ self.cooccurences

    def predict(self, X):
        # Sum up values of coocurrences
        X = X.tocsr()
        return X @ self.cooccurences


class MostPopular(Recommender):
    """ Most Popular """
    def __init__(self):
        self.most_popular = None

    def __str__(self):
        return "Most Popular baseline"

    def train(self, X):
        self.most_popular = X.tocsr().sum(0)

    def predict(self, X):
        return np.broadcast_to(self.most_popular, X.size())
