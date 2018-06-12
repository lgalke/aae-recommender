
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

from mpd import playlists_from_slices, unpack_playlists

DATA_PATH = "/data21/lgalke/MPD/data/"


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
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit the max number of considered items")
    args = parser.parse_args()

    # The mapping from command line arguments to create the model
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
    data = unpack_playlists(playlists)
    # data is: raw items, playlist ids, side info
    data = Bags(data[0], data[1], {"title": data[3]})

    print("Building vocabulary")
    data = data.build_vocab(max_features=args.limit, apply=True)

    print("Training...")
    model.train(data)

    # TODO load test set and apply model


if __name__ == '__main__':
    main()
