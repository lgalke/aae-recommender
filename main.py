import argparse
from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.baselines import RandomBaseline, Countbased, MostPopular
from aaerec.svd import SVDRecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
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
PARSER.add_argument('-e', '--epochs', type=int, default=50)
PARSER.add_argument('--lr', type=float, default=0.001)

ARGS = PARSER.parse_args()


DATASET = Bags.load_tabcomma_format(ARGS.dataset, unique=True)


EVAL = Evaluation(DATASET, ARGS.year, logfile=ARGS.outfile)
EVAL.setup(min_count=ARGS.min_count, min_elements=2)

print("Loading pre-trained embedding", W2V_PATH)
VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

BASELINES = [
    # RandomBaseline(),
    # MostPopular(),
    Countbased(),
    SVDRecommender(1000, use_title=False),
]

ae_params = {
    'n_code': 50,
    'n_epochs': 100,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}

vae_params = {
    'n_code': 50,
    # VAE results get worse with more epochs in preliminary optimization 
    #(Pumed with threshold 50)
    'n_epochs': 50,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}

RECOMMENDERS = [
    # AAERecommender(adversarial=False, lr=ARGS.lr, **ae_params),
    # AAERecommender(gen_lr=ARGS.lr, reg_lr=ARGS.lr, **ae_params),
    VAERecommender(conditions=None, **vae_params),
    DAERecommender(conditions=None, **ae_params)
]


CONDITIONS = ConditionList([
    ('title', PretrainedWordEmbeddingCondition(VECTORS))
])


CONDITIONED_MODELS = [
#    AAERecommender(adversarial=False,
#                   conditions=CONDITIONS,
#                   lr=ARGS.lr,
#                   **ae_params),
#    AAERecommender(adversarial=True,
#                   conditions=CONDITIONS,
#                   gen_lr=ARGS.lr,
#                   reg_lr=ARGS.lr,
#                   **ae_params),
#    DecodingRecommender(CONDITIONS,
#                        n_epochs=ARGS.epochs, batch_size=100, optimizer='adam',
#                        n_hidden=100, lr=ARGS.lr, verbose=True),
     VAERecommender(conditions=CONDITIONS, **vae_params),
     DAERecommender(conditions=CONDITIONS, **ae_params)
]


TITLE_ENHANCED = [
    SVDRecommender(1000, use_title=True),
    # DecodingRecommender(n_epochs=100, batch_size=100, optimizer='adam',
    #                     n_hidden=100, embedding=VECTORS,
    #                     lr=0.001, verbose=True),
    # AAERecommender(adversarial=False, use_title=True, lr=0.001,
    #                **ae_params),
    # AAERecommender(adversarial=True, use_title=True,
    #                prior='gauss', gen_lr=0.001, reg_lr=0.001,
    #                **ae_params),
]

with open(ARGS.outfile, 'a') as fh:
    print("~ Conditioned Models", "~" * 42, file=fh)
EVAL(RECOMMENDERS)
with open(ARGS.outfile, 'a') as fh:
    print("~ Conditioned Models", "~" * 42, file=fh)
EVAL(CONDITIONED_MODELS)
# with open(ARGS.outfile, 'a') as fh:
#     print("~ Partial List + Titles", "~" * 42, file=fh)
# EVAL(TITLE_ENHANCED)
