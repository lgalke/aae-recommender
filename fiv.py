"""
Executable to run AAE on the IREON dataset
"""
import argparse
import json

from aaerec.datasets import Bags
from mpd import log
from aaerec.evaluation import Evaluation
from aaerec.svd import SVDRecommender
from aaerec.baselines import Countbased
from aaerec.aae import AAERecommender, DecodingRecommender
from gensim.models.keyedvectors import KeyedVectors

# Should work on kdsrv03
DATA_PATH = "/data22/ivagliano/SWP/FivMetadata.json"
DEBUG_LIMIT = None
METRICS = ['mrr', 'map']

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True
VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

ae_params = {
    'n_code': 50,
    'n_epochs': 20,
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


def parse_en_labels(subjects):
    """
    From subjects in the json formats to a list of english descriptors of subjects
    """
    labels = []
    for subject in subjects:
        if subject["name_en"] != "":
            labels.append(subject["name_en"])

    return labels


def unpack_papers(papers):
    """
    Unpacks list of papers in a way that is compatible with our Bags dataset
    format. It is not mandatory that papers are sorted.
    """

    bags_of_labels, ids, side_info, years = [], [], {}, {}
    for paper in papers:
        # Extract ids
        ids.append(paper["id"])
        # Put all subjects assigned to the paper in here
        try:
            # Subject may be missing
            bags_of_labels.append(parse_en_labels(paper["subject"]))
        except KeyError:
            bags_of_labels.append([])
        # Use dict here such that we can also deal with unsorted ids
        try:
            side_info[paper["id"]] = paper["title"]
        except KeyError:
            side_info[paper["id"]] = ""
        try:
            years[paper["id"]] = paper["date"]
        except KeyError:
            years[paper["id"]] = -1

    # bag_of_labels and ids should have corresponding indices
    # In side_info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_labels, ids, {"title": side_info, "year": years}


def main(year, dataset, min_count=None, outfile=None):
    """ Main function for training and evaluating AAE methods on DBLP data """
    print("Loading data from", DATA_PATH)
    papers = load(DATA_PATH)
    print("Unpacking {} data...".format(dataset))
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
    parser.add_argument('year', type=int,
                        help='First year of the testing set.')
    parser.add_argument('-d', '--dataset', type=str,
                        help="Parse the DBLP or ACM dataset", default="dblp")
    parser.add_argument('-m', '--min-count', type=int,
                        help='Pruning parameter', default=50)
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.",
                        type=str, default=None)
    args = parser.parse_args()
    main(year=args.year, dataset=args.dataset, min_count=args.min_count, outfile=args.outfile)
