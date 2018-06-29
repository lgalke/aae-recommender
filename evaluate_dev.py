#!/usr/bin/env python3
import argparse

from mpd import playlists_from_slices, DATA_PATH
from mpd_metrics import aggregate_metrics


def load_submission(path):
    """ Returns dict of list: pid -> track_ids """
    sub = {}
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line[0] == '#' or line.startswith('team_info'):
                continue
            pid, *tracks = line.split(',')
            sub[int(pid)] = tracks
    return sub

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exclude', type=argparse.FileType('r'),
                        help="Path to exclude file, determines ground truth.")
    parser.add_argument('submission', type=str,
                        help="Path to dev submission file")
    parser.add_argument('-v', '--verbose', default=0, type=int)

    args = parser.parse_args()

    dev_slices = [line.strip() for line in args.exclude]
    if args.verbose:
        print("Loading ground truth from", dev_slices)

    ground_truth = playlists_from_slices(DATA_PATH, only=dev_slices, verbose=args.verbose)
    # Make the json stuff dictionaries from pid to track uris
    ground_truth = {p['pid']: [t['track_uri'] for t in p['tracks']] for p in ground_truth}

    predictions = load_submission(args.submission)


    # Verify that pids match
    pids = set(ground_truth.keys())
    pids_pred = set(predictions.keys())
    if not pids_pred:
        print(args.submission, 'is empty.')
        exit(1)
    if args.verbose:
        print(len(pids), "pids in ground truth")
        print(len(pids_pred), "pids in predictions")
        print(len(set.intersection(pids, pids_pred)), "pids in intersection")
    # Super strict: All pids in both are the same
    assert len(pids ^ pids_pred) == 0

    # Less strict: all predicted pids should be also in gold
    assert len(pids_pred - pids) == 0



    summary = aggregate_metrics(ground_truth, predictions,
                                500, pids)

    print(summary)








    args = parser.parse_args()



if __name__ == '__main__':
    main()
