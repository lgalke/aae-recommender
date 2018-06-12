
"""
Executable to make a submission using AAE on the Spotify MDP dataset
"""
import argparse
import glob
import itertools
import json
import os

from datasets import Bags
from baselines import Countbased
from aae import AAERecommender, DecodingRecommender
from svd import SVDRecommender

from mpd import playlists_from_slices, unpack_playlists, load

DATA_PATH = "/data21/lgalke/MPD/data/"
TEST_PATH = "/data21/lgalke/MPD/challenge_set.json"

SUBMISSION_HEADER = "team_info,Unconscious Bias,lga@informatik.uni-kiel.de"


def make_submission(predictions,
                    index2playlist,
                    index2trackid,
                    outfile=None,
                    topk=500):
    """ Writes the predictions as submission file to disk """
    # TODO sort topk items per playlist
    with open(outfile, 'w') as fhandle:
        print(SUBMISSION_HEADER, file=fhandle)
        # Line format
        # playlist_id, trackid1, trackid2, trackid500


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
    parser.add_argument('-o', '--outfile', default="submission.csv",
                        type=str, help="Write submissions to this path")
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

    print("Loading data...")
    playlists = playlists_from_slices(DATA_PATH)
    train_set = Bags(*unpack_playlists(playlists))

    print("Building vocabulary")
    vocab, __counts = train_set.build_vocab(max_features=args.max_items,
                                            apply=True)

    print("Training...")
    model.train(train_set)

    # Training finished, training set not necessary anymore
    del train_set
    print("Loading test set")
    data, index2playlist, side_info = unpack_playlists(load(TEST_PATH))
    test_set = Bags(data, index2playlist, side_info)
    # Apply same vocabulary as in training
    test_set.apply_vocab(vocab)

    pred = model.predict(test_set)
    # TODO: make non sparse
    # TODO: remove non-missing

    index2trackid = {v: k for k, v in vocab.items()}
    make_submission(pred, index2playlist, index2trackid, outfile=args.outfile)


if __name__ == '__main__':
    main()
