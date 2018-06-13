#!/usr/bin/env python3
"""
Executable to make a submission using AAE on the Spotify MDP dataset
"""
import argparse
import csv

import numpy as np
import scipy.sparse as sp

from datasets import Bags
from baselines import Countbased
from aae import AAERecommender, DecodingRecommender
from svd import SVDRecommender
from evaluation import remove_non_missing, argtopk

from mpd import playlists_from_slices, unpack_playlists, load
from mpd import TRACK_INFO

DATA_PATH = "/data21/lgalke/MPD/data/"
TEST_PATH = "/data21/lgalke/MPD/challenge_set.json"

SUBMISSION_HEADER = ["team_info", "Unconscious Bias",
                     "lga@informatik.uni-kiel.de"]


def make_submission(predictions,
                    index2playlist,
                    index2trackid,
                    outfile=None,
                    topk=500):
    """ Writes the predictions as submission file to disk """
    print("Sorting top {} items for each playlist".format(topk))
    __, topk_iy = argtopk(predictions, topk)
    print("Writing rows to", outfile)
    with open(outfile, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(SUBMISSION_HEADER)
        # Line format
        # playlist_id, trackid1, trackid2, trackid500
        for row_ix, item_ixs in enumerate(topk_iy):
            playlist = index2playlist[row_ix]
            items = [index2trackid[ix] for ix in item_ixs]
            csv_writer.writerow([playlist] + items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        # All possible method should appear here
                        choices=['cm', 'svd', 'ae', 'aae', 'mlp'],
                        help="Specify the model to use")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Specify the number of training epochs [50]")
    parser.add_argument('--hidden', type=int, default=100,
                        help="Number of hidden units [100]")
    parser.add_argument('--no-title', action='store_false', default=True,
                        dest='use_title',
                        help="Do not use the playlist titles")
    parser.add_argument('--max-items', type=int, default=None,
                        help="Limit the max number of considered items")
    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs for data loading [4].")
    parser.add_argument('-o', '--outfile', default="submission.csv",
                        type=str, help="Write submissions to this path")
    parser.add_argument('--aggregate', action='store_true', default=False,
                        help="Aggregate track metadata as side info input")
    args = parser.parse_args()

    # Create the model as specified by command line args
    # Count-based never uses title
    # Decoding recommender always uses title
    model = {
        'cm': Countbased(),
        'svd': SVDRecommender(use_title=args.use_title),
        'ae': AAERecommender(use_title=args.use_title,
                             adversarial=False,
                             n_hidden=args.hidden,
                             n_epochs=args.epochs),
        'aae': AAERecommender(use_title=args.use_title,
                              adversarial=True,
                              n_hidden=args.hidden,
                              n_epochs=args.epochs),
        'mlp': DecodingRecommender(n_epochs=args.epochs,
                                   n_hidden=args.hidden)
    }[args.model]

    track_attrs = TRACK_INFO if args.aggregate else None


    # = Training =
    print("Loading data from {} using {} jobs".format(DATA_PATH, args.jobs))
    playlists = playlists_from_slices(DATA_PATH, n_jobs=args.jobs)
    print("Unpacking playlists")
    train_set = Bags(*unpack_playlists(playlists, aggregate=track_attrs))

    print("Building vocabulary of {} most frequent items".format(args.max_items))
    vocab, __counts = train_set.build_vocab(max_features=args.max_items,
                                            apply=False)
    train_set = train_set.apply_vocab(vocab)
    print("Training set:", train_set, sep='\n')

    print("Training for {} epochs".format(args.epochs))
    try:
        model.train(train_set)
    except KeyboardInterrupt:
        print("Training interrupted by keyboard, pass.")

    # Not required anymore
    del train_set

    # = Predictions =
    print("Loading and unpacking test set")
    data, index2playlist, side_info = unpack_playlists(load(TEST_PATH),
                                                       aggregate=track_attrs)
    test_set = Bags(data, index2playlist, side_info)
    # Apply same vocabulary as in training
    test_set = test_set.apply_vocab(vocab)
    print("Test set:", test_set, sep='\n')

    pred = model.predict(test_set)
    if sp.issparse(pred):
        pred = pred.toarray()
    else:
        pred = np.asarray(pred)
    print("Scaling and removing non-missing items")
    pred = remove_non_missing(pred, test_set.tocsr(), copy=False)

    index2trackid = {v: k for k, v in vocab.items()}
    print("Making submission:", args.outfile)
    make_submission(pred, index2playlist, index2trackid, outfile=args.outfile)
    print("Success.")
    print("Make sure to verify the submission format with the provided script")


if __name__ == '__main__':
    main()
