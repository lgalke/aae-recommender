# -*- coding: utf-8 -*-
#
# Copyright 2017 Spotify AB.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Metrics for the 2018 RecSys Challenge. Includes:
  - R precision
  - Normalized discounted cumulative gain (NDCG)
  - Playlist extender clicks

All functions take as input:
  - targets: list of target track, artist... identifiers
  - predictions: SORTED list of predicted track, artist... identifiers; most relevant item comes first
Identifiers should be comparable through the '==' operator!
"""

__author__ = "Cedric De Boom"
__status__ = "beta"
__version__ = "0.1"
__date__ = "2017 October 27"

from collections import OrderedDict
from collections import namedtuple

import numpy as np
from scipy import stats


# R precision
def r_precision(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    target_set = set(targets)
    target_count = len(target_set)
    return float(len(set(predictions[:target_count]).intersection(target_set))) / target_count

def dcg(relevant_elements, retrieved_elements, k, *args, **kwargs):
    """Compute the Discounted Cumulative Gain.

    Rewards elements being retrieved in descending order of relevance.
    \[ DCG = rel_1 + \sum_{i=2}^{|R|} \frac{rel_i}{\log_2(i + 1)} \]

    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation

    Note: The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.

    Returns:
        DCG value

    """
    retrieved_elements = __get_unique(retrieved_elements[:k])
    relevant_elements = __get_unique(relevant_elements)
    if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
        return 0.0
    # Computes an ordered vector of 1.0 and 0.0
    score = [float(el in relevant_elements) for el in retrieved_elements]
    # return score[0] + np.sum(score[1:] / np.log2(
    #     1 + np.arange(2, len(score) + 1)))
    return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))


def ndcg(relevant_elements, retrieved_elements, k, *args, **kwargs):
    r"""Compute the Normalized Discounted Cumulative Gain.

    Rewards elements being retrieved in descending order of relevance.
    The metric is determined by calculating the DCG and dividing it by the
    ideal or optimal DCG in the case that all recommended tracks are relevant.

    Note:
    The ideal DCG or IDCG is on our case equal to:
    \[ IDCG = 1+\sum_{i=2}^{min(\left| G \right|, k)}\frac{1}{\log_2(i +1)}\]
    If the size of the set intersection of \( G \) and \( R \), is empty, then
    the IDCG is equal to 0. The NDCG metric is now calculated as:
    \[ NDCG = \frac{DCG}{IDCG + \delta} \]
    with \( \delta \) a (very) small constant.
    The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.

    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation

    Returns:
        NDCG value

    """

    # TODO: When https://github.com/scikit-learn/scikit-learn/pull/9951 is
    # merged...
    idcg = dcg(
        relevant_elements, relevant_elements, min(k, len(relevant_elements)))
    if idcg == 0:
        raise ValueError("relevent_elements is empty, the metric is"
                         "not defined")
    true_dcg = dcg(relevant_elements, retrieved_elements, k)
    return true_dcg / idcg


def __get_unique(original_list):
    """Get only unique values of a list but keep the order of the first
    occurence of each element
    """
    return list(OrderedDict.fromkeys(original_list))


Metrics = namedtuple('Metrics', ['r_precision', 'ndcg', 'plex_clicks'])


# playlist extender clicks
def playlist_extender_clicks(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    i = set(predictions).intersection(set(targets))
    for index, t in enumerate(predictions):
        for track in i:
            if t == track:
                return float(int(index / 10))
    return float(max_n_predictions / 10.0 + 1)


# def compute all metrics
def get_all_metrics(targets, predictions, k):
    return Metrics(r_precision(targets, predictions, k),
                   ndcg(targets, predictions, k),
                   playlist_extender_clicks(targets, predictions, k))


MetricsSummary = namedtuple('MetricsSummary', ['mean_r_precision',
                                               'mean_ndcg',
                                               'mean_plex_clicks',
                                               'coverage'])


def aggregate_metrics(ground_truth, sub, k, candidates):
    r_precision = []
    ndcg = []
    plex_clicks = []
    miss = 0
    cnt = 0
    for p in candidates:
        cnt += 1
        if p not in sub:
            miss += 1
            m = Metrics(0, 0, 0)  # TODO: make sure this is right
        else:
            m = get_all_metrics(ground_truth[p], sub[p], k)
        r_precision.append(m.r_precision)
        ndcg.append(m.ndcg)
        plex_clicks.append(m.plex_clicks)

    cov = 1 - miss / float(cnt)
    return MetricsSummary(
        stats.describe(r_precision).mean,
        stats.describe(ndcg).mean,
        stats.describe(plex_clicks).mean,
        cov
    )
