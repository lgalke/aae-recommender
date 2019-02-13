import scipy.sparse as sp

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import Recommender
from .ub import AutoEncoderMixin

class SVDRecommender(Recommender, AutoEncoderMixin):
    """ SVD Baseline, capable of dealing with text """
    def __init__(self, dims=1000, use_title=False, tfidf_params={}, **kwargs):
        super(SVDRecommender, self).__init__()
        if use_title:
            self.tfidf = TfidfVectorizer(input='content', **tfidf_params)

        self.svd = TruncatedSVD(dims, **kwargs)
        self.use_title = use_title

    def __str__(self):
        return str(self.svd)

    def fit(self, X, y=None):
        self.svd.fit(X)
        return self

    def transform(self, X, y=None):
        return self.svd.transform(X)

    def inverse_transform(self, X, y=None):
        return self.svd.inverse_transform(X)

    def train(self, training_set):
        x_train = training_set.tocsr()
        self.n_classes = x_train.shape[1]
        if self.use_title:
            titles = training_set.get_single_attribute("title")
            titles = self.tfidf.fit_transform(titles)
            x_train = sp.hstack([x_train, titles])
        self.fit(x_train)

    def predict(self, test_set):
        x_test = test_set.tocsr()
        if self.use_title:
            titles = test_set.get_single_attribute("title")
            titles = self.tfidf.transform(titles)
            x_test = sp.hstack([x_test, titles])
        # hidden = self.svd.transform(x_test)
        # y_pred = self.svd.inverse_transform(hidden)
        y_pred = self.reconstruct(x_test)
        y_pred = y_pred[:, :self.n_classes]
        return y_pred
