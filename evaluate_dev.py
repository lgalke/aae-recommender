#!/usr/bin/env python3
import argparse

from mpd import playlists_from_slices, DATA_PATH, load
from mpd_metrics import aggregate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exclude', type=argparse.FileType('r'),
                        help="Path to exclude file, determines ground truth.")
    parser.add_argument('submission', type=str,
                        help="Path to dev submission file")

    args = parser.parse_args()

    dev_slices = [line.split() for line in args.exclude]

    ground_truth = playlists_from_slices(DATA_PATH, only=dev_slices)
    predictions = load(args.submission)

    # Make the lists dictionaries based on pids
    ground_truth = {p['pid']: p for p in ground_truth}
    predictions  = {p['pid']: p for p in predictions}

    # Verify that pids match
    pids = set(ground_truth.keys())
    pids_pred = set(predictions.keys())
    #  no pids in one of them but not both
    assert len(pids ^ pids_pred) == 0
    # all predicted pids should be also in gold
    assert len(pids - pids_pred) == 0



    summary = aggregate_metrics(ground_truth, predictions,
                                500, pids)

    print(summary)








    args = parser.parse_args()



if __name__ == '__main__':
    main()
