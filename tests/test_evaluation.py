from aaerec.evaluation import evaluate
from aaerec.evaluation import MRR, MAP, P
import numpy as np
from sklearn.datasets import make_multilabel_classification

EPS = 1e-8


def test_batching():
    """ Test whether batched evaluation yields same results as non-batched """
    n_samples = 120
    n_classes = 10
    X, Y = make_multilabel_classification(n_samples, 20, n_classes)

    predictions = np.random.rand(n_samples, n_classes)
    metrics = [MRR(), MAP(), P(1), P(5)]

    results = evaluate(Y, predictions, metrics, batch_size=None)
    results_batched = evaluate(Y, predictions, metrics, batch_size=25)

    results_mean, results_std = zip(*results)
    results_batched_mean, results_batched_std = zip(*results_batched)

    assert ((np.array(results_batched_mean) - np.array(results_mean)) < EPS).all()
    assert ((np.array(results_batched_std) - np.array(results_std)) < EPS).all()



