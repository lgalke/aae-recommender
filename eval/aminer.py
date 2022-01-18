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

import aaerec.aae
aaerec.aae.USE_WANDB = True

from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.baselines import Countbased
from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.svd import SVDRecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition



# Import log from MPD causes static variables to be loaded (e.g. VECTORS)
# Instead I copied the log function
# from eval.mpd.mpd import log




def log(*print_args, logfile=None):
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        with open(logfile, 'a') as fhandle:
            print(*print_args, file=fhandle)
    print(*print_args)

# Set to a folder containing both ACM and DBLP datasets
# lga Feb 17, 2021: Script exec CWD is now assumed to be repository's root
if __name__ == '__main__':
    DATA_PATH = "./aminer/"
    DEBUG_LIMIT = None
    PAPER_INFO = ['title', 'venue', 'author']


    # Set to the word2vec-Google-News-corpus file
    # lga Feb 17, 2021: Script exec CWD is now assumed to be repository's root
    W2V_PATH = "./vectors/GoogleNews-vectors-negative300.bin.gz"
    W2V_IS_BINARY = True
    print("Loading keyed vectors")
    VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
    print("Done")

    # Hyperparameters
    AE_PARAMS = {
        'n_code': 50,
        'n_epochs': 20,
    #    'embedding': VECTORS,
        'batch_size': 10000,
        'n_hidden': 100,
        'normalize_inputs': True,
    }


# Metadata to use



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

    return list(papers)


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
    title_cnt = author_cnt = ref_cnt = venue_cnt = one_ref_cnt = 0
    for paper in papers:
        # Extract ids
        ids.append(paper["id"])
        # Put all ids of cited papers in here
        try:
            # References may be missing
            bags_of_refs.append(paper["references"])
            if len(paper["references"]) > 0:
                ref_cnt += 1
            if len(paper["references"]) == 1:
                one_ref_cnt += 1
        except KeyError:
            bags_of_refs.append([])
        # Use dict here such that we can also deal with unsorted ids
        try:
            side_info[paper["id"]] = paper["title"]
            if paper["title"] != "":
                title_cnt += 1
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

        try:
            if len(paper["authors"]) > 0:
                author_cnt += 1
        except KeyError:
            pass
        try:
            if len(paper["venue"]) > 0:
                venue_cnt += 1
        except KeyError:
            pass

        # We could assemble even more side info here from the track names
        if aggregate is not None:
            aggregated_paper_info = aggregate_paper_info(paper, aggregate)
            side_info[paper["id"]] += ' ' + aggregated_paper_info

    print("Metadata-fields' frequencies: references={}, title={}, authors={}, venue={}, one-reference={}"
          .format(ref_cnt/len(papers), title_cnt/len(papers), author_cnt/len(papers), venue_cnt/len(papers), one_ref_cnt/len(papers)))

    # bag_of_refs and ids should have corresponding indices
    # In side info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_refs, ids, {"title": side_info, "year": years, "author": authors, "venue": venue}


def main(year, dataset, min_count=None, outfile=None, drop=1,
        baselines=False,
        autoencoders=False,
        conditioned_autoencoders=False,
        all_metadata=True):
    """ Main function for training and evaluating AAE methods on DBLP data """

    assert baselines or autoencoders or conditioned_autoencoders, "Please specify what to run"


    if all_metadata:
        # V2 - all metadata
        CONDITIONS = ConditionList([
            ('title', PretrainedWordEmbeddingCondition(VECTORS)),
            ('venue', PretrainedWordEmbeddingCondition(VECTORS)),
            ('author', CategoricalCondition(embedding_dim=32, reduce="sum", # vocab_size=0.01,
                                            sparse=False, embedding_on_gpu=True))
        ])
    else:
        # V1 - only title metadata
        CONDITIONS = ConditionList([('title', PretrainedWordEmbeddingCondition(VECTORS))])
    #### CONDITOINS defined

    ALL_MODELS = []

    if baselines:
        # Models without metadata
        BASELINES = [
            # RandomBaseline(),
            # MostPopular(),
            Countbased(),
            SVDRecommender(1000, use_title=False)
        ]


        ALL_MODELS += BASELINES

        if not all_metadata:
            # SVD can use only titles not generic conditions
            ALL_MODELS += [SVDRecommender(1000, use_title=True)]

    if autoencoders:
        AUTOENCODERS = [
            AAERecommender(adversarial=False,
                           conditions=None,
                           lr=0.001,
                           **AE_PARAMS),
            AAERecommender(adversarial=True,
                           conditions=None,
                           gen_lr=0.001,
                           reg_lr=0.001,
                            **AE_PARAMS),
            VAERecommender(conditions=None, **AE_PARAMS),
            DAERecommender(conditions=None, **AE_PARAMS)
        ]
        ALL_MODELS += AUTOENCODERS

    if conditioned_autoencoders:
        # Model with metadata (metadata used as set in CONDITIONS above)
        CONDITIONED_AUTOENCODERS = [
            AAERecommender(adversarial=False,
                           conditions=CONDITIONS,
                           lr=0.001,
                           **AE_PARAMS),
            AAERecommender(adversarial=True,
                           conditions=CONDITIONS,
                           gen_lr=0.001,
                           reg_lr=0.001,
                            **AE_PARAMS),
            DecodingRecommender(CONDITIONS,
                                n_epochs=100, batch_size=1000, optimizer='adam',
                                n_hidden=100, lr=0.001, verbose=True),
            VAERecommender(conditions=CONDITIONS, **AE_PARAMS),
            DAERecommender(conditions=CONDITIONS, **AE_PARAMS)
        ]
        ALL_MODELS += CONDITIONED_AUTOENCODERS


    print("Finished preparing models:", *ALL_MODELS, sep='\n\t')


    path = DATA_PATH + ("dblp-ref/" if dataset == "dblp" else "acm.txt")
    print("Loading data from", path)
    papers = papers_from_files(path, dataset, n_jobs=4)
    print("Unpacking {} data...".format(dataset))
    bags_of_papers, ids, side_info = unpack_papers(papers)
    del papers
    bags = Bags(bags_of_papers, ids, side_info)
    if args.compute_mi:
        from aaerec.utils import compute_mutual_info
        print("[MI] Dataset:", dataset)
        print("[MI] min Count:", min_count)
        tmp = bags.build_vocab(min_count=min_count, max_features=None)
        mi = compute_mutual_info(tmp, conditions=None, include_labels=True,
                                  normalize=True)
        with open('mi.csv', 'a') as mifile:
            print(dataset, min_count, mi, sep=',', file=mifile)

        print("=" * 78)
        exit(0)

    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)

    evaluation = Evaluation(bags, year, logfile=outfile)
    evaluation.setup(min_count=min_count, min_elements=2, drop=drop)
    with open(outfile, 'a') as fh:
        print("~ Partial List + Titles + Author + Venue", "~" * 42, file=fh)
    evaluation(ALL_MODELS, batch_size=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int,
                        help='First year of the testing set.')
    parser.add_argument('-d', '--dataset', type=str,
                        help="Parse the DBLP or ACM dataset", default="dblp",
                        choices=["dblp", "acm"])
    parser.add_argument('-m', '--min-count', type=int,
                        help='Pruning parameter', default=None)
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.",
                        type=str, default=None)
    parser.add_argument('-dr', '--drop', type=str,
                        help='Drop parameter', default="1")
    parser.add_argument('--compute-mi', default=False,
                        action='store_true')
    parser.add_argument('--all_metadata', default=False,
                        action='store_true')
    parser.add_argument('--baselines', default=False,
                        action='store_true')
    parser.add_argument('--autoencoders', default=False,
                        action='store_true')
    parser.add_argument('--conditioned_autoencoders', default=False,
                        action='store_true')
    args = parser.parse_args()

    # Drop could also be a callable according to evaluation.py but not managed as input parameter
    try:
        drop = int(args.drop)
    except ValueError:
        drop = float(args.drop)

    main(year=args.year, dataset=args.dataset, min_count=args.min_count, outfile=args.outfile, drop=drop,
            all_metadata=args.all_metadata,
            baselines=args.baselines,
            autoencoders=args.autoencoders,
            conditioned_autoencoders=args.conditioned_autoencoders)
