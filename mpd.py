"""
Executable to run AAE on the Spotify Million Playlist Dataset
"""
import glob
import os
import json

from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from datasets import Bags, corrupt_sets
from transforms import lists2sparse
from evaluation import remove_non_missing, evaluate
from baselines import Countbased
from aae import AAERecommender

# Should work on kdsrv03
DATA_PATH = "/data21/lgalke/MDP/data/"

METRICS = ['mrr', 'map']

MODELS = [
    Countbased(),
    AAERecommender(adversarial=True, use_title=True)
    # Put more here...
]


def playlists_from_slices(slices_dir, sort_by_pid=True):
    """
    Loads a bunch of slices into a list of playlists,
    optionally sorted by id
    """
    playlists = []
    for fpath in glob.iglob(os.path.join(slices_dir, '*.json')):
        with open(fpath, 'r') as fhandle:
            data_slice = json.load(fhandle)
            playlists.extend(data_slice["playlists"])

    # in-place sort by playlist id
    if sort_by_pid:
        playlists.sort(key=itemgetter("pid"))

    return playlists


def unpack_playlists(playlists):
    """
    Unpacks list of playlists in a way that is compatible with our Bags dataset
    format. It is not mandatory that playlists are sorted.
    """
    # Assume track_uri is primary key for track
    bags_of_tracks = [[t["track_uri"] for t in p["tracks"]] for p in playlists]
    # Extract pids (even though they might be sorted already)
    pids = [p["pid"] for p in playlists]
    # Use dict here such that we can also deal with unsorted pids or other keys
    side_info = {p["pid"]: p["name"] for p in playlists}
    # We could assemble even more side info here from the track names and so on
    return bags_of_tracks, pids, side_info


def prepare_evaluation(bags, test_size=0.1):
    """
    Split data into train and dev set.
    Build vocab on train set and applies it to both train and test set.
    """
    # Split 10% validation data, one submission per day is too much.
    train_set, dev_set = bags.train_test_split(test_size=test_size)
    # Builds vocabulary only on training set
    # Limit of most frequent 50000 distinct items is for testing purposes
    vocab, __counts = train_set.build_vocab(max_features=50000)

    # Apply vocab (turn track ids into indices)
    train_set = train_set.apply_vocab(vocab)
    # Discard unknown tokens in the test set
    dev_set = train_set.apply_vocab(vocab)

    # Make sure that all playlists have two or more tracks left
    min_elements = 2
    train_set.prune_(min_elements)
    dev_set.prune_(min_elements)

    # Drop one track off each playlist within test set
    noisy, missing = corrupt_sets(dev_set.data, drop=1)
    assert len(noisy) == len(missing) == len(dev_set)
    # Replace test data with corrupted data
    dev_set.data = noisy

    return train_set, dev_set, missing


def main():
    """ Main function for training and evaluating AAE methods on MDP data """
    print("Loading data from", DATA_PATH)
    playlists = playlists_from_slices(DATA_PATH, sort_by_pid=False)
    print("Unpacking json data...")
    bags_of_tracks, pids, side_info = unpack_playlists(playlists)
    # Re-use 'title' property here because methods rely on it
    bags = Bags(bags_of_tracks, pids, {"title": side_info})
    print("Whole dataset:")
    print(bags)
    train_set, dev_set, missing = prepare_evaluation(bags)

    print("Retained items in train set:")
    print(train_set)

    print("Retained items in dev set:")
    print(dev_set)

    # THE GOLD
    y_test = lists2sparse(missing, dev_set.size(1)).tocsr(copy=False)

    # the known items in the test set, just to not recompute
    x_test = lists2sparse(dev_set.data, dev_set.size(1)).tocsr(copy=False)

    # These need to be implemented in evaluation.py

    for model in MODELS:
        print('=' * 78)
        print(model)

        # Training
        model.train(train_set)

        # Prediction
        y_pred = model.predict(dev_set)

        # Sanity-fix #1, make sparse stuff dense, expect array
        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()
        else:
            y_pred = np.asarray(y_pred)

        # Sanity-fix, remove predictions for already present items
        y_pred = remove_non_missing(y_pred, x_test, copy=False)

        # Evaluate metrics
        results = evaluate(y_test, y_pred, METRICS)

        print("-" * 78)
        for metric, stats in zip(METRICS, results):
            print("* {}: {} ({})".format(metric, *stats))

        print('=' * 78)


if __name__ == '__main__':
    main()
