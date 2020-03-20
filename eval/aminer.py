"""
Executable to run AAE on the AMiner DBLP dataset
"""
import argparse
import glob
import itertools
import json
import os

from gensim.models.keyedvectors import KeyedVectors
from joblib import Parallel, delayed

from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.baselines import Countbased
from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.svd import SVDRecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from eval.mpd.mpd import log

from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition

# Should work on kdsrv03
DATA_PATH = "/data22/ivagliano/aminer/"
DEBUG_LIMIT = None
# authors is an array of strings, n_citation is an integer: do they make sense used in this way?
PAPER_INFO = ['title', 'venue', 'author']
# TODO Add it as parameters of evaluate() to make it effective (see mpd.py)
# METRICS = ['mrr', 'map']

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True
print("Loading keyed vectors") 
VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
print("Done") 

ae_params = {
    'n_code': 50,
    'n_epochs': 20,
#    'embedding': VECTORS,
    'batch_size': 1000,
    'n_hidden': 100,
    'normalize_inputs': True,
}

BASELINES = [
    # RandomBaseline(),
    # MostPopular(),
    Countbased(),
    SVDRecommender(1000, use_title=False),
]

RECOMMENDERS = [
    # AAERecommender(use_title=False, adversarial=False, lr=0.0001,
    #                **ae_params),
    # AAERecommender(use_title=False, prior='gauss', gen_lr=0.0001,
    #                reg_lr=0.0001, **ae_params),
    # VAERecommender(conditions=None, **ae_params)
    # DAERecommender(conditions=None, **ae_params)
]

# TITLE_ENHANCED = [
#     SVDRecommender(1000, use_title=True),
#     DecodingRecommender(n_epochs=100, batch_size=100, optimizer='adam',
#                         n_hidden=100, embedding=VECTORS,
#                         lr=0.001, verbose=True),
#     AAERecommender(adversarial=False, use_title=True, lr=0.001,
#                    **ae_params),
#     AAERecommender(adversarial=True, use_title=True,
#                    prior='gauss', gen_lr=0.001, reg_lr=0.001,
#                    **ae_params),
# ]

CONDITIONS = ConditionList([
    ('title', PretrainedWordEmbeddingCondition(VECTORS)),
    ('venue', PretrainedWordEmbeddingCondition(VECTORS)),
    ('author', CategoricalCondition(embedding_dim=32, reduce="sum", # vocab_size=0.01,
                                    sparse=True, embedding_on_gpu=True))
])

CONDITIONED_MODELS = [
    AAERecommender(adversarial=False,
                   conditions=CONDITIONS,
                   lr=0.001,
                   **ae_params),
    AAERecommender(adversarial=True,
                   conditions=CONDITIONS,
                   gen_lr=0.001,
                   reg_lr=0.001,
                    **ae_params),
    DecodingRecommender(CONDITIONS,
                        n_epochs=100, batch_size=1000, optimizer='adam',
                        n_hidden=100, lr=0.001, verbose=True),
    VAERecommender(conditions=CONDITIONS, **ae_params),
    DAERecommender(conditions=CONDITIONS, **ae_params)
]


def load_dblp(path):
    """ Loads a single file """
    with open(path, 'r') as fhandle:
        obj = [json.loads(line.rstrip('\n')) for line in fhandle]
    return obj


def load_acm(path):
    """ Loads a single file """
    with open(path, 'r') as fhandle:
        obj = []
        paper = {}
        paper["references"] = []

        for line in fhandle:
            line = line.rstrip('\n')

            if len(line) == 0:
                obj.append(paper)
                paper = {}
                paper["references"] = []

            elif line[1] == '*':
                paper["title"] = line[2:]
            elif line[1] == '@':
                paper["authors"] = line[2:].split(",")
            elif line[1] == 't':
                paper["year"] = int(line[2:])
            elif line[1] == 'c':
                paper["venue"] = line[2:]
            elif line[1] == 'i':
                paper["id"] = line[6:]
            else:
                paper["references"].append(line[2:])

    return obj


def papers_from_files(path, dataset, n_jobs=1, debug=False):
    """
    Loads a bunch of files into a list of papers,
    optionally sorted by id
    """
    if dataset == "acm":
        return load_acm(path)

    it = glob.iglob(os.path.join(path, '*.json'))
    if debug:
        print("Debug mode: using only two slices")
        it = itertools.islice(it, 2)
    n_jobs = int(n_jobs)
    if n_jobs == 1:
        papers = []
        for i, fpath in enumerate(it):
            papers.extend(load_dblp(fpath))
            print("\r{}".format(i+1), end='', flush=True)
            if DEBUG_LIMIT and i > DEBUG_LIMIT:
                # Stop after `DEBUG_LIMIT` files
                # (for quick testing)
                break
        print()
    else:
        pps = Parallel(n_jobs=n_jobs, verbose=5)(delayed(load_dblp)(p) for p in it)
        papers = itertools.chain.from_iterable(pps)

    return papers


def aggregate_paper_info(paper, attributes):
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

    bags_of_refs, ids, side_info, years, authors, venue = [], [], {}, {}, {}, {}
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
        try:
            authors[paper["id"]] = paper["authors"]
        except KeyError:
            authors[paper["id"]] = []
        try:
            venue[paper["id"]] = paper["venue"]
        except KeyError:
            venue[paper["id"]] = ""

        # We could assemble even more side info here from the track names
        if aggregate is not None:
            aggregated_paper_info = aggregate_paper_info(paper, aggregate)
            side_info[paper["id"]] += ' ' + aggregated_paper_info

    # bag_of_refs and ids should have corresponding indices
    # In side info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_refs, ids, {"title": side_info, "year": years, "author": authors, "venue": venue}


def main(year, dataset, min_count=None, outfile=None, drop=1):
    """ Main function for training and evaluating AAE methods on DBLP data """
    path = DATA_PATH + ("dblp-ref/" if dataset == "dblp" else "acm.txt")
    print("Loading data from", path)
    papers = papers_from_files(path, dataset, n_jobs=4)
    print("Unpacking {} data...".format(dataset))
    bags_of_papers, ids, side_info = unpack_papers(papers)
    del papers
    bags = Bags(bags_of_papers, ids, side_info)

    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)

    evaluation = Evaluation(bags, year, logfile=outfile)
    evaluation.setup(min_count=min_count, min_elements=2, drop=drop)

    # with open(outfile, 'a') as fh:
    #     print("~ Partial List", "~" * 42, file=fh)
    # evaluation(BASELINES + RECOMMENDERS)
    # evaluation(RECOMMENDERS, batch_size=1000)

    with open(outfile, 'a') as fh:
        print("~ Partial List + Titles + Author + Venue", "~" * 42, file=fh)
    # evaluation(TITLE_ENHANCED)
    evaluation(CONDITIONED_MODELS, batch_size=1000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int,
                        help='First year of the testing set.')
    parser.add_argument('-d', '--dataset', type=str,
                        help="Parse the DBLP or ACM dataset", default="dblp")
    parser.add_argument('-m', '--min-count', type=int,
                        help='Pruning parameter', default=50)
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.",
                        type=str, default=None)
    parser.add_argument('-dr', '--drop', type=str,
                        help='Drop parameter', default="1")
    args = parser.parse_args()

    # Drop could also be a callable according to evaluation.py but not managed as input parameter
    try:
        drop = int(args.drop)
    except ValueError:
        drop = float(args.drop)

    main(year=args.year, dataset=args.dataset, min_count=args.min_count, outfile=args.outfile, drop=drop)
