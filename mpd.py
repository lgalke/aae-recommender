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
DATA_PATH = "/data21/lgalke/MPD/data/"
DEBUG_LIMIT = None
# Use only this many most frequent items
N_ITEMS = 50000
# Use all present items
# N_ITEMS = None

# These need to be implemented in evaluation.py
METRICS = ['mrr', 'map']

MODELS = [
    Countbased(),
    AAERecommender(adversarial=False, use_title=False, n_epochs=10)
    # Put more here...
]


def playlists_from_slices(slices_dir, sort_by_pid=False):
    """
    Loads a bunch of slices into a list of playlists,
    optionally sorted by id
    """
    playlists = []
    for i, fpath in enumerate(glob.iglob(os.path.join(slices_dir, '*.json'))):
        with open(fpath, 'r') as fhandle:
            data_slice = json.load(fhandle)
            playlists.extend(data_slice["playlists"])
        if DEBUG_LIMIT and i > DEBUG_LIMIT:
            # Stop after `DEBUG_LIMIT` files
            # (for quick testing)
            break

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


def prepare_evaluation(bags, test_size=0.1, n_items=None):
    """
    Split data into train and dev set.
    Build vocab on train set and applies it to both train and test set.
    """
    # Split 10% validation data, one submission per day is too much.
    train_set, dev_set = bags.train_test_split(test_size=test_size)
    # Builds vocabulary only on training set
    # Limit of most frequent 50000 distinct items is for testing purposes
    vocab, __counts = train_set.build_vocab(max_features=n_items,
                                            apply=False)

    # Apply vocab (turn track ids into indices)
    train_set = train_set.apply_vocab(vocab)
    # Discard unknown tokens in the test set
    dev_set = dev_set.apply_vocab(vocab)

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
    train_set, dev_set, y_test = prepare_evaluation(bags, n_items=N_ITEMS)

    print("Train set:")
    print(train_set)

    print("Dev set:")
    print(dev_set)

    # THE GOLD (put into sparse matrix)
    y_test = lists2sparse(y_test, dev_set.size(1)).tocsr(copy=False)

    # the known items in the test set, just to not recompute
    x_test = lists2sparse(dev_set.data, dev_set.size(1)).tocsr(copy=False)


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
