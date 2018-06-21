#!/usr/bin/env python3
import argparse
import os
import json
from collections import Counter
from datetime import date

import random

import numpy as np

from mpd import playlists_from_slices

# Ratio between named and nameless playlists
P_NAMELESS = 0.2

# distribution dicts of form n_retain: n_count
DIST_NAMELESS = {10: 1000, 5: 1000}
DIST_NAMED = {100: 2000, 25: 2000, 0: 1000, 1: 1000, 5: 1000, 10: 1000}

DATA_PATH = "/data21/lgalke/MPD/data"


def random_keep(tracks, dist):
    """
    Returns randomly sampled tracks according to dist and the number of
    holdouts
    """
    # Unzip distribution
    values, weights = zip(*dist.items())

    # Brute-force fix for too small playlists, might skew distribution a little
    # Always drop at least some...
    red_dist = [(val, p) for (val, p) in dist.items() if val < len(tracks)]
    values, weights = zip(*red_dist)
    # Renormalize such that weights sum up to 1
    weights_sum = sum(weights)
    weights = [w / weights_sum for w in weights]

    # Determine number of tracks to keep according to (reduced) dist
    keep = np.random.choice(values, p=weights)

    # in-place shuffle
    random.shuffle(tracks)
    retain = tracks[:keep]
    num_holdouts = len(tracks[keep:])
    if num_holdouts == 0:
        print("WARNING: no holdouts")
    return retain, num_holdouts


def corrupt_playlists(playlists):
    # Name-less playlists should have more than 10 tracks
    long_enough = [p for p in playlists if len(p) > max(DIST_NAMELESS.keys())]
    long_enough = []
    too_short = []
    len_threshold = max(DIST_NAMELESS.keys())

    # Split playlists into the ones that are long enough for nameless
    # treatment and the ones that are too short
    for playlist in playlists:
        if len(playlist['tracks']) > len_threshold:
            long_enough.append(playlist)
        else:
            too_short.append(playlist)

    dev_playlists = []

    # Random sample n_nameless among long enough ones
    random.shuffle(long_enough)
    n_nameless = int(P_NAMELESS * len(playlists))
    for playlist in long_enough[:n_nameless]:
        pcopy = {k: v for (k, v) in playlist.items()}
        del pcopy['name']
        retained_tracks, num_holdouts = random_keep(pcopy['tracks'],
                                                    DIST_NAMELESS)
        pcopy['tracks'] = retained_tracks
        pcopy['num_holdouts'] = num_holdouts
        dev_playlists.append(pcopy)

    # Merge remaining and too short playlists
    remainder = long_enough[n_nameless:] + too_short

    # Treat named, in place
    for playlist in remainder:
        pcopy = {k: v for (k, v) in playlist.items()}
        retained_tracks, num_holdouts = random_keep(pcopy['tracks'],
                                                    DIST_NAMED)
        pcopy['tracks'] = retained_tracks
        pcopy['num_holdouts'] = num_holdouts
        dev_playlists.append(pcopy)

    assert len(dev_playlists) == len(playlists)

    return dev_playlists


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('slices', type=argparse.FileType('r'),
                        help="Path to file with one slice filename per line")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="File to put output")
    args = parser.parse_args()

    if args.output is None:
        print("No output file specified, performing a dry run")

    if os.path.exists(args.output) and \
            input("Path '{}' exists. Overwrite? [y/N]"
                  .format(args.output)) != 'y':
        exit(-1)

    # strip newlines from exclude names
    slices = [s.strip() for s in args.slices]
    print("Creating dev set from slices:", slices)

    playlists = playlists_from_slices(DATA_PATH, only=slices)

    dev_playlists = corrupt_playlists(playlists)

    dev_set = {
        'date': str(date.today()),
        'version': 'dev set created from: ' + str(slices),
        'playlists': dev_playlists
    }

    with open(args.output, 'w') as fhandle:
        json.dump(dev_set, fhandle)

if __name__ == '__main__':
    main()
