"""
Executable to run AAE on the IREON dataset
"""
import argparse
import json
import re

from gensim.models.keyedvectors import KeyedVectors

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
DATA_PATH = "/data22/ivagliano/SWP/FivMetadata.json"
CLEAN_DATA_PATH = "/data22/ivagliano/SWP/FivMetadata_clean.json"
CLEAN = False
DEBUG_LIMIT = None
METRICS = ['mrr', 'map']

W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True
print("Loading pre-trained embedding", W2V_PATH)
VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
print("Done")

ae_params = {
    'n_code': 50,
    'n_epochs': 20,
    # 'embedding': VECTORS,
    'batch_size': 500,
    'n_hidden': 100,
    'normalize_inputs': True,
}

vae_params = {
    'n_code': 50,
    # VAE results get worse with more epochs in preliminary optimization 
    #(Pumed with threshold 50)
    'n_epochs': 50,
    'batch_size': 500,
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
    AAERecommender(adversarial=False, lr=0.001,
                   **ae_params),
    AAERecommender(prior='gauss', gen_lr=0.001,
                   reg_lr=0.001, **ae_params),
    VAERecommender(conditions=None, **vae_params),
    DAERecommender(conditions=None, **ae_params)
]

CONDITIONS = ConditionList([
    ('title', PretrainedWordEmbeddingCondition(VECTORS)),
#    ('author', CategoricalCondition(embedding_dim=32, reduce="sum",
#                                    sparse=True, embedding_on_gpu=True))
])

CONDITIONED_MODELS = [
    # TODO SVD can use only titles not generic conditions
    SVDRecommender(1000, use_title=True),
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
                       n_epochs=20, batch_size=500, optimizer='adam',
                        n_hidden=100, lr=0.001, verbose=True),
    VAERecommender(conditions=CONDITIONS, **vae_params),
    DAERecommender(conditions=CONDITIONS, **ae_params)
]


def load(path):
    """ Loads a single file """
    with open(path, 'r') as fhandle:
        obj = [json.loads(line.rstrip('\n')) for line in fhandle]
    return obj


def clean(path, papers):
    with open(path, "w") as write_file:
        for p in papers:
            try:
                p["year"] = p.pop("date")
            except KeyError:
                continue
            p["subjects"] = parse_en_labels(p.pop("subject"))
            p["authors"] = parse_authors(p)
            if len(p["year"]) < 4:
                continue
            if len(p["year"]) >= 4:
                matches = re.findall(r'.*([1-2][0-9]{3})', p["year"])
                # if no or more than one match skip string
                if len(matches) == 0 or len(matches) > 1:
                    print("no match for {}".format(p["year"]))
                    continue
                else:
                    try:
                        p["year"] = int(matches[0])
                    except ValueError:
                        print("Value error for {}".format(matches[0]))
                        continue
            if (p["year"] > 2016):
                continue
            write_file.write(json.dumps(p) + "\n")


def parse_en_labels(subjects):
    """
    From subjects in the json formats to a list of english descriptors of subjects
    """
    labels = []
    for subject in subjects:
        if subject["name_en"] != "":
            labels.append(subject["name_en"])

    return labels

def parse_authors(p):
    """
    From Marc21-IDs in the json formats to a list of authors
    """
    authors = []
    try:
        for creator in p.pop("creator_personal"):
            authors.append(creator.pop("name"))
    except KeyError:
        pass

    for obj in p.pop("Marc21-IDs"):
        try:
            author = obj.pop("700").pop("entry")
        except KeyError:
            continue
        if "Author aut" in author:
           authors.append(author.replace(", Author aut", ""))

    return authors


def unpack_papers_conditions(papers):
    """
    Unpacks list of papers in a way that is compatible with our Bags dataset
    format. It is not mandatory that papers are sorted.
    """

    bags_of_labels, ids, side_info, years, authors = [], [], {}, {}, {}
    for paper in papers:
        # Extract ids
        ids.append(paper["id"])
        # Put all subjects assigned to the paper in here
        try:
            # Subject may be missing
            bags_of_labels.append(paper["subjects"])
        except KeyError:
            bags_of_labels.append([])

        # Use dict here such that we can also deal with unsorted ids
        try:
            side_info[paper["id"]] = paper["title"]
        except KeyError:
            side_info[paper["id"]] = ""
        try:
            years[paper["id"]] = paper["year"]
        except KeyError:
            years[paper["id"]] = -1

        authors[paper["id"]] = paper["authors"]

    # bag_of_labels and ids should have corresponding indices
    # In side_info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_labels, ids, {"title": side_info, "year": years, "author": authors}


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
            bags_of_labels.append(paper["subjects"])
        except KeyError:
            bags_of_labels.append([])
        # Use dict here such that we can also deal with unsorted ids
        try:
            side_info[paper["id"]] = paper["title"]
        except KeyError:
            side_info[paper["id"]] = ""
        try:
            years[paper["id"]] = paper["year"]
        except KeyError:
            years[paper["id"]] = -1

    # bag_of_labels and ids should have corresponding indices
    # In side_info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_labels, ids, {"title": side_info, "year": years}


def main(year, min_count=None, outfile=None, drop=1):
    """ Main function for training and evaluating AAE methods on IREON data """
    if (CLEAN == True):
        print("Loading data from", DATA_PATH)
        papers = load(DATA_PATH)
        print("Cleaning data...")
        clean(CLEAN_DATA_PATH, papers)
        print("Clean data in {}".format(CLEAN_DATA_PATH))
        return

    print("Loading data from", CLEAN_DATA_PATH)
    papers = load(CLEAN_DATA_PATH)
    print("Unpacking IREON data...")
    # bags_of_papers, ids, side_info = unpack_papers(papers)
    bags_of_papers, ids, side_info = unpack_papers_conditions(papers)
    del papers
    bags = Bags(bags_of_papers, ids, side_info)

    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)

    evaluation = Evaluation(bags, year, logfile=outfile)
    evaluation.setup(min_count=min_count, min_elements=2, drop=drop)

    # Use only partial citations/labels list (no additional metadata)
    with open(outfile, 'a') as fh:
        print("~ Partial List", "~" * 42, file=fh)
    evaluation(BASELINES + RECOMMENDERS)
    # Use additional metadata (as defined in CONDITIONS for all models but SVD, which uses only titles)
    with open(outfile, 'a') as fh:
        print("~ Conditioned Models", "~" * 42, file=fh)
    evaluation(CONDITIONED_MODELS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int,
                        help='First year of the testing set.')
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

    main(year=args.year, min_count=args.min_count, outfile=args.outfile, drop=drop)
