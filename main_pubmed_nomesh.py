"""
Executable to run AAE on the PubMed (CITREC) and Econis datasets
- For PubMed models can use no metadata, just titles and titles + more metadata
- For Econis models can use no metadata and titles
- To run the models on Econis using titles + more metadata see the separate script /eval/econis.py
"""
import argparse
import os
from collections import OrderedDict

from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.baselines import RandomBaseline, Countbased, MostPopular
from aaerec.svd import SVDRecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from gensim.models.keyedvectors import KeyedVectors

from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition

# Set this to the word2vec Google News corpus file
W2V_PATH = "./vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True

# Command line arguments
PARSER = argparse.ArgumentParser()
PARSER.add_argument('dataset', type=str,
                    help='path to dataset')
PARSER.add_argument('year', type=int,
                    help='First year of the testing set.')
PARSER.add_argument('-m', '--min-count', type=int,
                    help='Pruning parameter', default=50)
PARSER.add_argument('-o', '--outfile', type=str, default=None)
PARSER.add_argument('-dr', '--drop', type=str,
                    help='Drop parameter', default="1")

ARGS = PARSER.parse_args()

# Drop could also be a callable according to evaluation.py but not managed as input parameter
try:
    drop = int(ARGS.drop)
except ValueError:
    drop = float(ARGS.drop)

# Only with more metadata (generic conditions) for Pubmed (Econis thorugh separate script /eval/econis.py)
# key: name of a table
# owner_id: ID of citing paper
# fields: list of column names in table
# target names: key for these data in the owner_attributes dictionary
# path: absolute path to the csv file
PMC_DATA_PATH = "/media/nvme1n1/lgalke/datasets/AAEREC/pmc_final_data"
mtdt_dic = OrderedDict()
mtdt_dic["author"] = {"owner_id": "pmId", "fields": ["name"],"target_names": ["author"],
                      "path": os.path.join(PMC_DATA_PATH, "author.csv")}
# No need to even load those
# mtdt_dic["mesh"] = {"owner_id": "document", "fields": ["descriptor"], "target_names": ["mesh"],
#                     "path": os.path.join(PMC_DATA_PATH, "mesh.csv")}

# With no metadata or just titles
# DATASET = Bags.load_tabcomma_format(ARGS.dataset, unique=True)
# With more metadata for PubMed (generic conditions)
DATASET = Bags.load_tabcomma_format(ARGS.dataset, unique=True, owner_str="pmId",
                                    set_str="cited", meta_data_dic=mtdt_dic)

EVAL = Evaluation(DATASET, ARGS.year, logfile=ARGS.outfile)
EVAL.setup(min_count=ARGS.min_count, min_elements=2, drop=drop)

print("Loading pre-trained embedding", W2V_PATH)
VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

# Hyperparameters
ae_params = {
    'n_code': 50,
    'n_epochs': 100,
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

# Models without metadata
BASELINES = [
    # RandomBaseline(),
    # MostPopular(),
    Countbased(),
    SVDRecommender(100, use_title=False)
]
RECOMMENDERS = [
    AAERecommender(adversarial=False, lr=0.001, **ae_params),
    AAERecommender(gen_lr=0.001, reg_lr=0.001, **ae_params),
    VAERecommender(conditions=None, **vae_params),
    DAERecommender(conditions=None, **ae_params)
]

# Metadata to use (apart for SVD, which uses only titles)
CONDITIONS = ConditionList([
    ('title', PretrainedWordEmbeddingCondition(VECTORS)),
    ('journal', CategoricalCondition(embedding_dim=32, reduce="sum",
                                     sparse=False, embedding_on_gpu=True)),
    ('author', CategoricalCondition(embedding_dim=32, reduce="sum",
                                    sparse=False, embedding_on_gpu=True))
#     ('mesh', CategoricalCondition(embedding_dim=32, reduce="sum",
#                                   sparse=True, embedding_on_gpu=True))
])

# Model with metadata (metadata used as set in CONDITIONS above)
CONDITIONED_MODELS = [
    # SVD can use only titles not generic conditions
    SVDRecommender(1000, use_title=True),
    AAERecommender(adversarial=False, conditions=CONDITIONS, **ae_params),
    AAERecommender(adversarial=True, conditions=CONDITIONS, **ae_params),
    DecodingRecommender(conditions=CONDITIONS, n_epochs=100, batch_size=500,
                           optimizer='adam',n_hidden=100, lr=0.001, verbose=True),
    VAERecommender(conditions=CONDITIONS, **vae_params),
    DAERecommender(conditions=CONDITIONS, **ae_params)
]


# Use only partial citations/labels list (no additional metadata)
with open(ARGS.outfile, 'a') as fh:
    print("~ Partial List", "~" * 42, file=fh)
EVAL(BASELINES + RECOMMENDERS)
# Use only additional metadata (as defined in CONDITIONS for all models but SVD, which uses only titles)
with open(ARGS.outfile, 'a') as fh:
    print("~ Conditioned Models", "~" * 42, file=fh)
EVAL(CONDITIONED_MODELS)
