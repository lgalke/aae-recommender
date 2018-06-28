"""
Executable to run AAE on the AMiner DBLP dataset
"""
import argparse
import glob
import itertools
import json
import os

# from joblib import Parallel, delayed

from datasets import Bags, corrupt_sets
from mpd import log
from evaluation import Evaluation
from svd import SVDRecommender
from baselines import Countbased
from aae import AAERecommender, DecodingRecommender
from gensim.models.keyedvectors import KeyedVectors

# Should work on kdsrv03
#DATA_PATH = "/data21/lgalke/MPD/data/"
DATA_PATH = "data/"
DEBUG_LIMIT = None
# Use only this many most frequent items
N_ITEMS = 50000
# Use all present items
# N_ITEMS = None
# authors is an array of strings, n_citation is an integer: do they make sense used in this way?
PAPER_INFO = ['title', 'venue', 'abstract']
METRICS = ['mrr', 'map']

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True
VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

ae_params = {
    'n_code': 50,
    'n_epochs': 100,
    'embedding': VECTORS,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}

BASELINES = [
    # RandomBaseline(),
    # MostPopular(),
    Countbased(),
    SVDRecommender(1000, use_title=False),
]

ae_params = {
    'n_code': 50,
    'n_epochs': 100,
    'embedding': VECTORS,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}

RECOMMENDERS = [
    AAERecommender(use_title=False, adversarial=False, lr=0.0001,
                   **ae_params),
    AAERecommender(use_title=False, prior='gauss', gen_lr=0.0001,
                   reg_lr=0.0001, **ae_params),
]

TITLE_ENHANCED = [
    SVDRecommender(1000, use_title=True),
    DecodingRecommender(n_epochs=100, batch_size=100, optimizer='adam',
                        n_hidden=100, embedding=VECTORS,
                        lr=0.001, verbose=True),
    AAERecommender(adversarial=False, use_title=True, lr=0.001,
                   **ae_params),
    AAERecommender(adversarial=True, use_title=True,
                   prior='gauss', gen_lr=0.001, reg_lr=0.001,
                   **ae_params),
]


def load(path):
    """ Loads a single file """
    with open(path, 'r') as fhandle:
        obj = [json.loads(line.rstrip('\n')) for line in fhandle]
    return obj


def papers_from_files(slices_dir, n_jobs=1, debug=False):
    """
    Loads a bunch of files into a list of papers,
    optionally sorted by id
    """
    it = glob.iglob(os.path.join(slices_dir, '*.json'))
    if debug:
        print("Debug mode: using only two slices")
        it = itertools.islice(it, 2)
    n_jobs = int(n_jobs)
    if n_jobs == 1:
        papers = []
        for i, fpath in enumerate(it):
            papers.extend(load(fpath))
            print("\r{}".format(i+1), end='', flush=True)
            if DEBUG_LIMIT and i > DEBUG_LIMIT:
                # Stop after `DEBUG_LIMIT` files
                # (for quick testing)
                break
        print()
    else:
        pps = Parallel(n_jobs=n_jobs, verbose=5)(delayed(load)(p) for p in it)
        papers = itertools.chain.from_iterable(pps)

    return papers


def aggregate_paper_info(paper, attributes):
    if 'tracks' not in paper:
        return ''
    acc = []
    for attribute in attributes:
        if attribute in paper:
            acc.append(paper[attribute])
    return ' '.join(acc)


def unpack_papers(papers, aggregate=None):
    """
    Unpacks list of papers in a way that is compatible with our Bags dataset
    format. It is not mandatory that papers are sorted.
    """
    # Assume track_uri is primary key for track
    if aggregate is not None:
        for attr in aggregate:
            assert attr in PAPER_INFO

    bags_of_refs, ids, side_info, years = [], [], {}, {}
    for paper in papers:
        # Extract ids
        ids.append(paper["id"])
        # Put all ids of cited papers in here
        try:
            # References may be missing
            bags_of_refs.append(paper["references"])
        except KeyError:
            bags_of_refs.append([])
        # Use dict here such that we can also deal with unsorted ids
        try:
            side_info[paper["id"]] = paper["title"]
        except KeyError:
            side_info[paper["id"]] = ""
        try:
            years[paper["id"]] = paper["year"]
        except KeyError:
            years[paper["id"]] = -1

        # We could assemble even more side info here from the track names
        if aggregate is not None:
            aggregated_paper_info = aggregate_paper_info(paper, aggregate)
            side_info[paper["id"]] += ' ' + aggregated_paper_info

    # bag_of_refs and ids should have corresponding indices
    # In side info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_refs, ids, {"title": side_info, "year": years}


def main(year=None, min_count=None, outfile=None):
    """ Main function for training and evaluating AAE methods on DBLP data """
    print("Loading data from", DATA_PATH)
    papers = papers_from_files(DATA_PATH, n_jobs=-1)
    print("Unpacking json data...")
    bags_of_papers, ids, side_info = unpack_papers(papers)
    del papers
    bags = Bags(bags_of_papers, ids, side_info)

    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)

    evaluation = Evaluation(bags, year, logfile=outfile)
    evaluation.setup(min_count=min_count, min_elements=2)
    print("Loading pre-trained embedding", W2V_PATH)

    with open(outfile, 'a') as fh:
        print("~ Partial List", "~" * 42, file=fh)
    evaluation(BASELINES + RECOMMENDERS)

    with open(outfile, 'a') as fh:
        print("~ Partial List + Titles", "~" * 42, file=fh)
    evaluation(TITLE_ENHANCED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.")
    parser.add_argument('year', type=int,
                        help='First year of the testing set.')
    parser.add_argument('-m', '--min-count', type=int,
                        help='Pruning parameter', default=50)
    parser.add_argument('-o', '--outfile', type=str, default=None)
    args = parser.parse_args()
    main(year=args.year, min_count=args.min_count, outfile=args.outfile)