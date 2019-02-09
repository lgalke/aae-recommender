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
    'n_epochs': 10,
    # 'embedding': VECTORS, # This belongs to condition now
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}

RECOMMENDERS = [
    AAERecommender(adversarial=False, lr=0.0001, **ae_params),
    AAERecommender(prior='gauss', gen_lr=0.0001, reg_lr=0.0001, **ae_params),
]


CONDITIONS = ConditionList([
    ('title', PretrainedWordEmbeddingCondition(VECTORS))
])


CONDITIONED_MODELS = [
    AAERecommender(adversarial=False, conditions=CONDITIONS, **ae_params),
    AAERecommender(adversarial=True, conditions=CONDITIONS, **ae_params),
    DecodingRecommender(CONDITIONS,
                        n_epochs=10, batch_size=100, optimizer='adam',
                        n_hidden=100, lr=0.001, verbose=True),
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
    print("~ Partial List", "~" * 42, file=fh)
EVAL(CONDITIONED_MODELS)
with open(ARGS.outfile, 'a') as fh:
    print("~ Partial List + Titles", "~" * 42, file=fh)
EVAL(TITLE_ENHANCED)
