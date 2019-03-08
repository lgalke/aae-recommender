"""
Executable to run AAE on the Spotify Million Playlist Dataset

Run via:

`python3 eval/mpd/mpd.py -o logfile.txt`

"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from gensim.models.keyedvectors import KeyedVectors

# Imports are broken, you can quickfix via symlink
# cd eval/mpd/; ln -s ../../aaerec aaerec

from aaerec.datasets import Bags, corrupt_sets
from aaerec.transforms import lists2sparse
from aaerec.evaluation import remove_non_missing, evaluate
from aaerec.baselines import Countbased
from aaerec.svd import SVDRecommender
from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition


DEBUG_LIMIT = None
# Use only this many most frequent items
N_ITEMS = None
# Use only items that appear this many times
MIN_COUNT = 55
# Use command line arg '-m' instead

TRACK_INFO = ['artist_name', 'track_name', 'album_name']
# TODO: find the side info fields
PLAYLIST_INFO = ['name']

#TFIDF_PARAMS = { 'max_features': N_WORDS }

SERVER = True

if SERVER:
    W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
    W2V_IS_BINARY = True
    VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
    DATA_PATH = "/data21/lgalke/datasets/MPD/data/"
    # DATA_PATH = "/data22/ggerstenkorn/citation_test_data/"
    CONDITIONS = ConditionList([
        ('name', PretrainedWordEmbeddingCondition(VECTORS))
    ])
else:

    print("load WE from file")
    W2V_PATH = "/workData/generalUseData/GoogleNews-vectors-negative300.bin.gz"
    W2V_IS_BINARY = True
    VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
    print("finished loading")

    DATA_PATH = "/workData/zbw/citation/local_data"
    #CONDITIONS = None
    CONDITIONS = ConditionList([
        ('name', PretrainedWordEmbeddingCondition(VECTORS)) # first element is name of attribute in  the dataset
    ])






# These need to be implemented in evaluation.py
METRICS = ['mrr']


MODELS = [
    AAERecommender(adversarial=False, conditions=CONDITIONS,n_epochs=1),
    # Only item sets
    #Countbased(),
    #SVDRecommender(1000, use_title=False),
    #AAERecommender(adversarial=True, use_title=False, n_epochs=55, embedding=VECTORS),
    #AAERecommender(adversarial=False, n_epochs=1),
    # Title-enhanced
    #SVDRecommender(1000, use_title=True),
    #AAERecommender(adversarial=True, use_side_info=True, n_epochs=55, embedding=VECTORS),
    #AAERecommender(adversarial=False, use_side_info=["name"], n_epochs=5, embedding=VECTORS),
    #DecodingRecommender(n_epochs=55, embedding=VECTORS)
    # Put more here...
]


def load(path):
    """ Loads a single slice """
    with open(path, 'r') as fhandle:
        obj = json.load(fhandle)
    return obj["playlists"]


def playlists_from_slices(slices_dir, n_jobs=1, debug=False, only=None, without=None, verbose=5):
    """
    Loads a bunch of slices into a list of playlists,
    optionally sorted by id

    :param slices_dir:
    :param n_jobs:
    :param debug:
    :param only:
    :param without:
    :param verbose:
    :return:
    """

    it = glob.glob(os.path.join(slices_dir, '*.json'))

    # Stuff to deal with dev set penc
    if only:
        it = [path for path in it if os.path.split(path)[1] in only]
    if without:
        it = [path for path in it if os.path.split(path)[1] not in without]

    if debug:
        print("Debug mode: using only two slices")
        it = it[:2]

    if verbose: 
        print("Loading", len(it), "slices using", n_jobs, "jobs.")
    n_jobs = int(n_jobs)
    if n_jobs == 1:
        playlists = []
        for i, fpath in enumerate(it):
            playlists.extend(load(fpath))
            if verbose:
                print("\r{}".format(i+1), end='', flush=True)
            if DEBUG_LIMIT and i > DEBUG_LIMIT:
                # Stop after `DEBUG_LIMIT` files
                # (for quick testing)
                break
        if verbose:
            print()
    else:
        pps = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(load)(p) for p in it)
        playlists = itertools.chain.from_iterable(pps)

    return playlists


def aggregate_track_info(playlist, attributes):
    """

    :param playlist: dict, one playlist instance with it's information
    :param attributes: iterable, keys of 'tracks' in playlist
    :return: str, bag of words all side info combined
    """
    if 'tracks' not in playlist:
        return ''
    acc = []
    for track in playlist['tracks']:
        for attribute in attributes:
            if attribute in track:
                acc.append(track[attribute])
    return ' '.join(acc)


def unpack_playlists(playlists, aggregate=None):
    """
    Unpacks list of playlists in a way that is compatible with our Bags dataset
    format. It is not mandatory that playlists are sorted.
    """
    # Assume track_uri is primary key for track
    if aggregate is not None:
        for attr in aggregate:
            assert attr in TRACK_INFO

    bags_of_tracks, pids, side_info = [], [], {}
    for playlist in playlists:
        # Extract pids
        pids.append(playlist["pid"])
        # Put all tracks of the playlists in here
        bags_of_tracks.append([t["track_uri"] for t in playlist["tracks"]])
        # Use dict here such that we can also deal with unsorted pids
        try:
            side_info[playlist["pid"]] = playlist["name"]
        except KeyError:
            side_info[playlist["pid"]] = ""

        # We could assemble even more side info here from the track names
        if aggregate is not None:
            aggregated_track_info = aggregate_track_info(playlist, aggregate)
            side_info[playlist["pid"]] += ' ' + aggregated_track_info

    # bag_of_tracks and pids should have corresponding indices
    # In side info the pid is the key
    # Re-use 'title' property here because methods rely on it
    return bags_of_tracks, pids, {"title": side_info}



def unpack_playlists_for_models_concatenated(playlists):
    """
    Unpacks list of playlists in a way that makes them ready for the models .train step.
    It is not mandatory that playlists are sorted.
    :param playlists: a dictionary, of playlists
    :param aggregate: an iterable, of potential names in the track model name space
    :param condition_name: a string, side info name, which to retrieve
    :return:
    """
    # Assume track_uri is primary key for track

    condition_names = PLAYLIST_INFO + TRACK_INFO
    print(condition_names)
    bags_of_tracks, pids = [], []
    side_infos = {condition:{} for condition in condition_names}
    for playlist in playlists:
        # Extract pids
        pids.append(playlist["pid"])
        # Put all tracks of the playlists in here
        bags_of_tracks.append([t["track_uri"] for t in playlist["tracks"]])
        # Use dict here such that we can also deal with unsorted pids



        for condition in condition_names:
            if condition in PLAYLIST_INFO:
                # stored: self.owner_attributes = side_info
                # called: self.owner_attributes[attribute][owner] # owner == pid
                # before: side_info[playlist["pid"]] = playlist["name"]
                # ordering doesn't matter as it's always called with pid together
                extracted_condition = playlist[condition] # whats coming out of playlist here? a string
                # TODO: think about more efficient handling via numpy/pandas in Bag class through slicing availability
            else:
                # TODO: add it in doctex
                enlisted_track_info = []
                for track in playlist["tracks"]:
                    enlisted_track_info.append(track[condition])
                extracted_condition = " ".join(enlisted_track_info)

            side_infos[condition][playlist["pid"]] = extracted_condition



    # for attr in side_infos:
    #     print(attr)
    #     for pid in list(side_infos[attr].keys())[:3]:
    #         print(pid,side_infos[attr][pid])

    # bag_of_tracks and pids should have corresponding indices
    # In side info the pid is the key
    # Re-use 'title' property here because methods rely on it
    return bags_of_tracks, pids, side_infos



def prepare_evaluation(bags, test_size=0.1, n_items=None, min_count=None):
    """
    Split data into train and dev set.
    Build vocab on train set and applies it to both train and test set.
    """
    # Split 10% validation data, one submission per day is too much.
    train_set, dev_set = bags.train_test_split(test_size=test_size)
    # Builds vocabulary only on training set
    # Limit of most frequent 50000 distinct items is for testing purposes
    vocab, __counts = train_set.build_vocab(max_features=n_items,
                                            min_count=min_count,
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


def log(*print_args, logfile=None):
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        with open(logfile, 'a') as fhandle:
            print(*print_args, file=fhandle)
    print(*print_args)


def main(outfile=None, min_count=None):
    """ Main function for training and evaluating AAE methods on MDP data """
    print("Loading data from", DATA_PATH)
    playlists = playlists_from_slices(DATA_PATH, n_jobs=4)
    print("Unpacking json data...")
    bags_of_tracks, pids, side_info = unpack_playlists_for_models_concatenated(playlists)
    del playlists
    bags = Bags(data=bags_of_tracks, owners=pids, owner_attributes=side_info)
    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)
    train_set, dev_set, y_test = prepare_evaluation(bags,
                                                    n_items=N_ITEMS,
                                                    min_count=min_count)

    log("Train set:", logfile=outfile)
    log(train_set, logfile=outfile)

    log("Dev set:", logfile=outfile)
    log(dev_set, logfile=outfile)

    # THE GOLD (put into sparse matrix)
    y_test = lists2sparse(y_test, dev_set.size(1)).tocsr(copy=False)

    # the known items in the test set, just to not recompute
    x_test = lists2sparse(dev_set.data, dev_set.size(1)).tocsr(copy=False)

    for model in MODELS:
        log('=' * 78, logfile=outfile)
        log(model, logfile=outfile)
        log(model.model_params, logfile=outfile)

        # Training
        model.train(train_set)
        print("training finished")

        # Prediction
        y_pred = model.predict(dev_set)
        print("prediction finished")

        print(" prediction sparse?:", sp.issparse(y_pred))
        # Sanity-fix #1, make sparse stuff dense, expect array
        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()
        else:
            y_pred = np.asarray(y_pred)

        print("remove non-missing:")
        # Sanity-fix, remove predictions for already present items
        y_pred = remove_non_missing(y_pred, x_test, copy=False)

        print("evaluate:")
        # Evaluate metrics
        results = evaluate(y_test, y_pred, METRICS, batch_size=100)

        print("metrics: ")
        log("-" * 78, logfile=outfile)
        for metric, stats in zip(METRICS, results):
            log("* {}: {} ({})".format(metric, *stats), logfile=outfile)

        log('=' * 78, logfile=outfile)


if __name__ == '__main__':

    # python3 ../eval/mpd/mpd.py -m 55 -o /data22/ivagliano/cit2vec-journal-results/mpd/titles-only/mpd-55-3.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.")
    parser.add_argument('-m', '--min-count', type=int,
                        default=None,
                        help="Minimum count of items")
    parser.add_argument('-s', '--side_information', type=str,
                        # TODO: handle more than one argument
                        default="name",
                        nargs='+',
                        help="list of incorporated additional attributes")
    args = parser.parse_args()
    print(args)
    main(outfile=args.outfile, min_count=args.min_count)
