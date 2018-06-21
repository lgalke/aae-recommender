import argparse
from collections import Counter

import numpy as np

from mpd import load

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('jsonfile')

    args = parser.parse_args()

    playlists = load(args.jsonfile)

    print("N =", len(playlists))

    lens = [len(p['tracks']) for p in playlists]

    print("Playlist track count:", Counter(lens))

    has_playlist = ['name' in p for p in playlists]
    print("Has playlist name:", Counter(has_playlist))

    nameless_lens = [len(p['tracks']) for p in playlists if 'name' not in p]
    print("Playlist track count among nameless:", Counter(nameless_lens))

    named_lens = [len(p['tracks']) for p in playlists if 'name' in p]
    print("Playlist track count among nameless:", Counter(named_lens))


    try:
        holdouts = np.array([p['num_holdouts'] for p in playlists])
        print("Holdouts: {:.2f} {:.2f}".format(holdouts.mean(), holdouts.std()))
    except KeyError:
        print("[warn] Num holdouts property missing")






if __name__ == '__main__':
    main()
