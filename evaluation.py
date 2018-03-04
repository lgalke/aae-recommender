from timeit import default_timer as timer
from datetime import timedelta
import os
import random
import sys
from abc import ABC, abstractmethod
from sklearn.preprocessing import minmax_scale
import numpy as np
import scipy.sparse as sp
import rank_metrics_with_std as rm
from datasets import corrupt_sets
from transforms import lists2sparse


def argtopk(X, k):
    """
    Picks the top k elements of (sparse) matrix X

    >>> X = np.arange(10).reshape(1, -1)
    >>> i = argtopk(X, 3)
    >>> X[argtopk(X, 3)]
    array([[9, 8, 7]])
    >>> X = np.arange(20).reshape(2,10)
    >>> X[argtopk(X, 3)]
    array([[ 9,  8,  7],
           [19, 18, 17]])
    >>> X = np.arange(6).reshape(2,3)
    >>> X[argtopk(X, 123123)]
    array([[2, 1, 0],
           [5, 4, 3]])
    """
    assert len(X.shape) == 2, "X should be two-dimensional array-like"
    rows = np.arange(X.shape[0])[:, np.newaxis]
    if k is None or k >= X.size:
        ind = np.argsort(X, axis=1)[:, ::-1]
        return rows, ind

    assert k > 0, "k should be positive integer or None"


    ind = np.argpartition(X, -k, axis=1)[:, -k:]
    # sort indices depending on their X values
    cols = ind[rows, np.argsort(X[rows, ind], axis=1)][:, ::-1]
    return rows, cols


class Metric(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass


class RankingMetric(Metric):
    """ Base class for all ranking metrics
    may also be used on its own to quickly get ranking scores from Y_true,
    Y_pred pair
    """

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', None)
        super().__init__()

    def __call__(self, y_true, y_pred):
        """ Gets relevance scores,
        Sort based on y_pred, then lookup in y_true
        >>> Y_true = np.array([[1,0,0],[0,0,1]])
        >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
        >>> RankingMetric(k=2)(Y_true, Y_pred)
        array([[0, 1],
               [1, 0]])
        """
        ind = argtopk(y_pred, self.k)
        rs = y_true[ind]
        return rs


class MRR(RankingMetric):
    """ Mean reciprocal rank at k

    >>> mrr_at_5 = MRR(5)
    >>> callable(mrr_at_5)
    True
    >>> Y_true = np.array([[1,0,0],[0,0,1]])
    >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
    >>> MRR(2)(Y_true, Y_pred)
    (0.75, 0.25)
    >>> Y_true = np.array([[1,0,1],[1,0,1]])
    >>> Y_pred = np.array([[0.4,0.3,0.2],[0.4,0.3,0.2]])
    >>> MRR(3)(Y_true, Y_pred)
    (1.0, 0.0)
    """
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred):
        # compute mrr wrt k
        rs = super().__call__(y_true, y_pred)
        return rm.mean_reciprocal_rank(rs)


class MAP(RankingMetric):
    """ Mean average precision at k """
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred):
        """
        >>> Y_true = np.array([[1,0,0],[0,0,1]])
        >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
        >>> MAP(2)(Y_true, Y_pred)
        (0.75, 0.25)
        >>> Y_true = np.array([[1,0,1],[1,0,1]])
        >>> Y_pred = np.array([[0.3,0.2,0.3],[0.6,0.5,0.7]])
        >>> MAP(3)(Y_true, Y_pred)
        (1.0, 0.0)
        >>> Y_true = np.array([[1,0,1],[1,1,1]])
        >>> Y_pred = np.array([[0.4,0.3,0.2],[0.4,0.3,0.2]])
        >>> MAP(3)(Y_true, Y_pred)
        (0.91666666666666663, 0.08333333333333337)
        """
        rs = super().__call__(y_true, y_pred)
        return rm.mean_average_precision(rs)


class P(RankingMetric):
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred):
        """
        >>> Y_true = np.array([[1,0,1,0],[1,0,1,0]])
        >>> Y_pred = np.array([[0.2,0.3,0.1,0.05],[0.2,0.5,0.7,0.05]])
        >>> P(2)(Y_true, Y_pred)
        (0.5, 0.0)
        >>> P(4)(Y_true, Y_pred)
        (0.5, 0.0)
        """
        # compute p wrt k
        rs = super().__call__(y_true, y_pred)
        ps = (rs > 0).mean(axis=1)
        return ps.mean(), ps.std()

BOUNDED_METRICS = {
    # (bounded) ranking metrics
    '{}@{}'.format(M.__name__.lower(), k): M(k)
    for M in [MRR, MAP, P] for k in [5, 10, 20]
}
BOUNDED_METRICS['P@1'] = P(1)


UNBOUNDED_METRICS = {
    # unbounded metrics
    M.__name__.lower(): M()
    for M in [MRR, MAP]
}

METRICS = { **BOUNDED_METRICS, **UNBOUNDED_METRICS }


def remove_non_missing(Y_pred, X_test, copy=True):
    """
    >>> Y_pred = np.array([[0.6,0.5,-1], [40,-20,10]])
    >>> X_test = np.array([[1, 0, 1], [0, 1, 0]])
    >>> remove_non_missing(Y_pred, X_test)
    array([[ 0.    ,  0.9375,  0.    ],
           [ 1.    ,  0.    ,  0.5   ]])
    """
    Y_pred_scaled = minmax_scale(Y_pred,
                                 feature_range=(0, 1),
                                 axis=1,  # Super important!
                                 copy=copy)
    # we remove the ones that were already present in the orig set
    Y_pred_scaled[X_test.nonzero()] = 0.
    return Y_pred_scaled


def evaluate(ground_truth, predictions, metrics):
    """
    Main evaluation function, used by Evaluation class but can also be
    reused to recompute metrics
    """
    if sp.issparse(predictions):
        predictions = predictions.toarray()
    if sp.issparse(ground_truth):
        ground_truth = ground_truth.toarray()

    metrics = [m if callable(m) else METRICS[m] for m in metrics]
    results = [metric(ground_truth, predictions) for metric in metrics]
    return results


def reevaluate(gold_file, predictions_file, metrics):
    """ Recompute metrics from files """
    Y_test = sp.load_npz(gold_file)
    Y_pred = np.load(predictions_file)
    return evaluate(Y_test, Y_pred, metrics)


def maybe_open(logfile, mode='a'):
    """
    If logfile is something that can be opened, do so else return STDOUT
    """
    return open(logfile, mode) if logfile else sys.stdout


def maybe_close(log_fh):
    """ Close if log_fh is not STDOUT """
    if log_fh is not sys.stdout:
        log_fh.close()


class Evaluation(object):
    def __init__(self,
                 dataset,
                 year,
                 metrics=METRICS,
                 logfile=sys.stdout,
                 logdir=None):
        self.dataset = dataset
        self.year = year
        self.metrics = metrics
        self.logfile = logfile
        self.logdir = logdir

        self.train_set, self.test_set = None, None
        self.x_test, self.y_test = None, None

    def setup(self, seed=42, min_elements=1, max_features=None,
              min_count=None, drop=1):
        # we could specify split criterion and drop choice here
        """ Splits and corrupts the data accordion to criterion """
        log_fh = maybe_open(self.logfile)
        random.seed(seed)
        np.random.seed(seed)
        # train_set, test_set = self.dataset.split(self.split_test,
        #                                          self.split_train)
        train_set, test_set = self.dataset.train_test_split(on_year=self.year)
        print("=" * 80, file=log_fh)
        print("Train:", train_set, file=log_fh)
        print("Test:", test_set, file=log_fh)
        print("Next Pruning:\n\tmin_count: {}\n\tmax_features: {}\n\tmin_elements: {}"
              .format(min_count, max_features, min_elements), file=log_fh)
        train_set = train_set.build_vocab(min_count=min_count,
                                          max_features=max_features,
                                          apply=True)
        test_set = test_set.apply_vocab(train_set.vocab)
        # Train and test sets are now BagsWithVocab
        train_set.prune_(min_elements=min_elements)
        test_set.prune_(min_elements=min_elements)
        print("Train:", train_set, file=log_fh)
        print("Test:", test_set, file=log_fh)
        print("Drop parameter:", drop)

        noisy, missing = corrupt_sets(test_set.data, drop=drop)

        assert len(noisy) == len(missing) == len(test_set)

        test_set.data = noisy
        print("-" * 80, file=log_fh)
        maybe_close(log_fh)

        # THE GOLD
        self.y_test = lists2sparse(missing, test_set.size(1)).tocsr(copy=False)

        self.train_set = train_set
        self.test_set = test_set

        # just store for not recomputing the stuff
        self.x_test = lists2sparse(noisy, train_set.size(1)).tocsr(copy=False)
        return self

    def __call__(self, recommenders):
        if None in (self.train_set, self.test_set, self.x_test, self.y_test):
            raise UserWarning("Call .setup() before running the experiment")

        if self.logdir:
            os.makedirs(self.logdir, exist_ok=True)
            vocab_path = os.path.join(self.logdir, "vocab.txt")
            with open(vocab_path, 'w') as vocab_fh:
                print(*self.train_set.index2token, sep='\n', file=vocab_fh)
            gold_path = os.path.join(self.logdir, "gold")
            sp.save_npz(gold_path, self.y_test)

        for recommender in recommenders:
            log_fh = maybe_open(self.logfile)
            print(recommender, file=log_fh)
            maybe_close(log_fh)
            train_set = self.train_set.clone()
            test_set = self.test_set.clone()
            t_0 = timer()
            # DONE FIXME copy.deepcopy is not enough!
            recommender.train(train_set)
            log_fh = maybe_open(self.logfile)
            print("Training took {} seconds."
                  .format(timedelta(seconds=timer()-t_0)), file=log_fh)

            t_1 = timer()
            y_pred = recommender.predict(test_set)
            if sp.issparse(y_pred):
                y_pred = y_pred.toarray()
            else:
                # dont hide that we are assuming an ndarray to be returned
                y_pred = np.asarray(y_pred)

            # set likelihood of documents that are already cited to zero, so
            # they don't influence evaluation
            y_pred = remove_non_missing(y_pred, self.x_test, copy=True)

            print("Prediction took {} seconds."
                  .format(timedelta(seconds=timer()-t_1)), file=log_fh)

            if self.logdir:
                t_1 = timer()
                pred_file = os.path.join(self.logdir, repr(recommender))
                np.save(pred_file, y_pred)
                print("Storing predictions took {} seconds."
                      .format(timedelta(seconds=timer()-t_1)), file=log_fh)

            t_1 = timer()
            results = evaluate(self.y_test, y_pred, metrics=self.metrics)
            print("Evaluation took {} seconds."
                  .format(timedelta(seconds=timer()-t_1)), file=log_fh)

            print("\nResults:\n", file=log_fh)
            for metric, (mean, std) in zip(self.metrics, results):
                print("- {}: {} ({})".format(metric, mean, std),
                      file=log_fh)
            print("\nOverall time: {} seconds."
                  .format(timedelta(seconds=timer()-t_0)), file=log_fh)
            print('-' * 79, file=log_fh)
            maybe_close(log_fh)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
