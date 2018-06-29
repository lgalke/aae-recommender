""" UB General Purpose Library """
from sklearn.feature_extraction.text import TfidfVectorizer


class AutoEncoderMixin(object):
    """ Mixin class for all sklearn-like Autoencoders """

    def reconstruct(self, X, y=None):
        """ Transform data, then inverse transform it """
        hidden = self.transform(X)
        return self.inverse_transform(hidden)


def peek_word2vec_format(path, binary=False):
    """
    Function to peek at the first line of a serialized embedding in
    word2vec format

    Arguments
    ---------
    path: The path to the file to peek
    binary: Whether the file is gzipped

    Returns
    -------
    Tuple of ints split by white space in the first line,
    i.e., for word2vec format the dimensions of the embedding.
    """
    if binary:
        import gzip
        with gzip.open(path, 'r') as peek:
            return map(int, next(peek).strip().split())
    else:
        with open(path, 'r') as peek:
            return map(int, next(peek).strip().split())


class EmbeddedVectorizer(TfidfVectorizer):

    """ Weighted Bag-of-embedded-Words"""

    def __init__(self, embedding, index2word, **kwargs):
        """
        Arguments
        ---------

        embedding: V x D embedding matrix
        index2word: list of words with indices matching V
        """
        super(EmbeddedVectorizer, self).__init__(self, vocabulary=index2word,
                                                 **kwargs)
        self.embedding = embedding

    def fit(self, raw_documents, y=None):
        super(EmbeddedVectorizer, self).fit(raw_documents)
        return self

    def transform(self, raw_documents, __y=None):
        sparse_scores = super(EmbeddedVectorizer,
                              self).transform(raw_documents)
        # Xt is sparse counts
        return sparse_scores @ self.embedding

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents, y)


class GensimEmbeddedVectorizer(EmbeddedVectorizer):
    """
    Shorthand to create an embedded vectorizer using a gensim KeyedVectors
    object, such that the vocabularies match.
    """

    def __init__(self, gensim_vectors, **kwargs):
        """
        Arguments
        ---------
        `gensim_vectors` is expected to have index2word and syn0 defined
        """
        index2word = gensim_vectors.index2word
        embedding = gensim_vectors.syn0
        super(GensimEmbeddedVectorizer, self).__init__(embedding,
                                                       index2word,
                                                       **kwargs)
