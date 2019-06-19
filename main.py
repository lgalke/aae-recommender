import argparse
from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.baselines import RandomBaseline, Countbased, MostPopular
from aaerec.svd import SVDRecommender
from gensim.models.keyedvectors import KeyedVectors

from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition
W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True


PARSER = argparse.ArgumentParser()
PARSER.add_argument('dataset', type=str,
                    help='path to dataset')
PARSER.add_argument('year', type=int,
                    help='First year of the testing set.')
PARSER.add_argument('-m', '--min-count', type=int,
                    help='Pruning parameter', default=50)
PARSER.add_argument('-o', '--outfile', type=str, default=None)

ARGS = PARSER.parse_args()

import os
from collections import OrderedDict
mtdt_dic = OrderedDict()
# tables["document"] = {"join_cit": "pmId", "join_mtdt": "pmId", "fields":["title","year", "month", "journal"]} # afterwards pmId of document is usable
mtdt_dic["author"] = {"owner_id": "pmId", "fields": ["id"],
                    "path": os.path.join("/data22/ggerstenkorn/citation_data_preprocessing/zbw_citation_preprocessing/",
                                         "author.csv")}
mtdt_dic["mesh"] = {"owner_id": "document", "fields": ["id"],
                    "path": os.path.join("/data22/ggerstenkorn/citation_data_preprocessing/zbw_citation_preprocessing/",
                                         "mesh.csv")}


DATASET = Bags.load_tabcomma_format(ARGS.dataset, unique=True,owner_str="pmId",set_str="cited", meta_data_dic=mtdt_dic)


EVAL = Evaluation(DATASET, ARGS.year, logfile=ARGS.outfile)
EVAL.setup(min_count=ARGS.min_count, min_elements=2)

print("Loading pre-trained embedding", W2V_PATH)
VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

BASELINES = [
    # RandomBaseline(),
    # MostPopular(),
    Countbased(),
    SVDRecommender(100, use_title=False),
]

ae_params = {
    'n_code': 5,
    'n_epochs': 5,
    # 'embedding': VECTORS, # This belongs to condition now
    'batch_size': 100,
    'n_hidden': 50,
    'normalize_inputs': True,
}

RECOMMENDERS = [
    AAERecommender(adversarial=False, lr=0.0001, **ae_params),
    AAERecommender(prior='gauss', gen_lr=0.0001, reg_lr=0.0001, **ae_params),
]


CONDITIONS = ConditionList([
    ('title', PretrainedWordEmbeddingCondition(VECTORS))
])

CONDITIONS_COMPLEX = ConditionList([
    ('title', PretrainedWordEmbeddingCondition(VECTORS))
    ('journal', CategoricalCondition(VECTORS,reduce=None))
    ('author', CategoricalCondition(VECTORS,reduce="sum"))
    ('mesh', CategoricalCondition(VECTORS,reduce="sum"))
])

CONDITIONED_MODELS = [
    AAERecommender(adversarial=False, conditions=CONDITIONS),
    AAERecommender(adversarial=True, conditions=CONDITIONS)
]


TITLE_ENHANCED = [
    SVDRecommender(100, use_title=True),
    # DecodingRecommender(n_epochs=100, batch_size=100, optimizer='adam',
    #                     n_hidden=100, embedding=VECTORS,
    #                     lr=0.001, verbose=True),
    AAERecommender(adversarial=False, use_title=True, lr=0.001,
                   **ae_params),
    AAERecommender(adversarial=True, use_title=True,
                   prior='gauss', gen_lr=0.001, reg_lr=0.001,
                   **ae_params),
]

with open(ARGS.outfile, 'a') as fh:
    print("~ Partial List", "~" * 42, file=fh)
EVAL(BASELINES + RECOMMENDERS + CONDITIONED_MODELS)
with open(ARGS.outfile, 'a') as fh:
    print("~ Partial List + Titles", "~" * 42, file=fh)
EVAL(TITLE_ENHANCED)
