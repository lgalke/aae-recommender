#!/usr/bin/env python3
"""
Executable to make a submission using AAE on the Spotify MDP dataset
"""
import argparse
import os
import csv

import numpy as np
import scipy.sparse as sp
from gensim.models.keyedvectors import KeyedVectors

from aaerec.datasets import Bags
from aaerec.baselines import Countbased
from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.svd import SVDRecommender
from aaerec.evaluation import remove_non_missing, argtopk

from mpd import playlists_from_slices, unpack_playlists, load
from mpd import TRACK_INFO

MPD_BASE_PATH = "/data21/lgalke/MPD"

DATA_PATH = os.path.join(MPD_BASE_PATH, "data")
TEST_PATH = os.path.join(MPD_BASE_PATH, "challenge_set.json")
VERIFY_SCRIPT = os.path.join(MPD_BASE_PATH, "verify_submission.py")

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True

SUBMISSION_HEADER = ["team_info",
                     "Unconscious Bias",
                     "main",
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
    with open(outfile, 'a') as csvfile:
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
    parser.add_argument('--model', type=str, defaule='aae',
                        # All possible method should appear here
                        choices=['cm', 'svd', 'ae', 'aae', 'mlp'],
                        help="Specify the model to use [aae]")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Specify the number of training epochs [50]")
    parser.add_argument('--hidden', type=int, default=200,
                        help="Number of hidden units [100]")
    parser.add_argument('--no-title', action='store_false', default=True,
                        dest='use_title',
                        help="Do not use the playlist titles")
    parser.add_argument('--max-items', type=int, default=75000,
                        help="Limit the max number of considered items")
    parser.add_argument('--vocab-size', type=int, default=50000,
                        help="Limit the max number of distinct condition words")
    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help="Number of jobs for data loading [4].")
    parser.add_argument('-o', '--outfile', default="submission.csv",
                        type=str, help="Write submissions to this path")
    parser.add_argument('--use-embedding', default=False, action='store_true',
                        help="Use embedding (SGNS GoogleNews) [false]")
    parser.add_argument('--dont-aggregate', action='store_false', dest='aggregate', default=True,
                        help="Aggregate track metadata as side info input")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="Activate debug mode, run only on small sample")
    parser.add_argument('-x', '--exclude', type=argparse.FileType('r'),  default=None,
                        help="Path to file with slice filenames to exclude for training")
    parser.add_argument('--dev', type=str, default=None,
                        help='Path to dev set, use in combination with (-x, --exclude)')
    parser.add_argument('--no-idf', action='store_false', default=True,
                        dest='use_idf', help="Do **not** use idf re-weighting")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Initial learning rate [0.001]")
    parser.add_argument('--code', type=int, default=100,
                        help="Code dimension [50]")
    args = parser.parse_args()

    # Either exclude and dev set, or no exclude and test set
    assert (args.dev is None) == (args.exclude is None)
    if args.dev is not None:
        print("Making submission for dev set:", args.dev)
        assert os.path.isfile(args.dev)


    # Dump args into submission file
    if os.path.exists(args.outfile) and \
            input("Path '{}' exists. Overwrite? [y/N]"
                  .format(args.outfile)) != 'y':
        exit(-1)

    with open(args.outfile, 'w') as out:
        print('#', args, file=out)

    print("Loading embedding:", W2V_PATH)
    if args.use_embedding:
        vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
    else:
        vectors = None

    # Create the model as specified by command line args
    # Count-based never uses title
    # Decoding recommender always uses title

    tfidf_params = {
        'max_features': args.vocab_size,
        'use_idf': args.use_idf
    }

    model = {
        'cm': Countbased(),
        'svd': SVDRecommender(use_title=args.use_title),
        'ae': AAERecommender(use_title=args.use_title,
                             adversarial=False,
                             n_hidden=args.hidden,
                             n_code=args.code,
                             n_epochs=args.epochs,
                             embedding=vectors,
                             lr=args.lr,
                             tfidf_params=tfidf_params),
        'aae': AAERecommender(use_title=args.use_title,
                              adversarial=True,
                              n_hidden=args.hidden,
                              n_code=args.code,
                              n_epochs=args.epochs,
                              gen_lr=args.lr,
                              reg_lr=args.lr, # same gen and reg lrs
                              embedding=vectors,
                              tfidf_params=tfidf_params),
        'mlp': DecodingRecommender(n_epochs=args.epochs,
                                   n_hidden=args.hidden,
                                   embedding=vectors,
                                   tfidf_params=tfidf_params)
    }[args.model]

    track_attrs = TRACK_INFO if args.aggregate else None

    if args.exclude is not None:
        # Dev set case, exclude dev set data
        exclude = [line.strip() for line in args.exclude]
    else:
        # Real submission case, do not exclude any training data
        exclude = None

    # = Training =
    print("Loading data from {} using {} jobs".format(DATA_PATH, args.jobs))
    playlists = playlists_from_slices(DATA_PATH, n_jobs=args.jobs, debug=args.debug,
                                      without=exclude)
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
    if args.dev is not None:
        print("Loading and unpacking DEV set")
        data, index2playlist, side_info = unpack_playlists(load(args.dev),
                                                           aggregate=track_attrs)
    else:
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
    print("Make sure to verify the submission format via", VERIFY_SCRIPT)


if __name__ == '__main__':
    main()
