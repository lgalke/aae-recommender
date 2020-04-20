"""
Executable to run AAE on the Reuters RCV1 Dataset

Run via:

`python3 eval/rcv.py -m <min_count> -o logfile.txt`

"""
import argparse

import numpy as np
import scipy.sparse as sp


# Imports are broken, you can quickfix via symlink
# cd eval/mpd/; ln -s ../../aaerec aaerec

from aaerec.datasets import Bags, corrupt_sets
from aaerec.transforms import lists2sparse
from aaerec.evaluation import remove_non_missing, evaluate
from aaerec.baselines import Countbased
from aaerec.svd import SVDRecommender
from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from gensim.models.keyedvectors import KeyedVectors
from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition

# Should work on kdsrv03
DATA_PATH = "/data22/ivagliano/Reuters/rcv1.tsv"
DEBUG_LIMIT = None

# These need to be implemented in evaluation.py
METRICS = ['mrr', 'map']
# W2V_PATH = "/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz"
# W2V_IS_BINARY = True

# print("Loading pre-trained embedding", W2V_PATH)
# VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

ae_params = {
    'n_code': 50,
    'n_epochs': 100,
    # 'embedding': VECTORS,
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

# CONDITIONS = ConditionList([
#     ('title', PretrainedWordEmbeddingCondition(VECTORS))
# ])

MODELS = [
    # Countbased(),  # Only item sets
    # SVDRecommender(10, use_title=False),
    # AAERecommender(adversarial=False, use_title=False, lr=0.001,
    #                **ae_params),
    # AAERecommender(adversarial=True, use_title=False, prior='gauss', gen_lr=0.001,
    #                reg_lr=0.001, **ae_params),
    VAERecommender(conditions=None, **vae_params),
    DAERecommender(conditions=None, **ae_params),
    # Title-enhanced
    # SVDRecommender(10, use_title=True),
    # AAERecommender(adversarial=False, use_title=True, lr=0.001,
    #                **ae_params),
    # AAERecommender(adversarial=True, use_title=True, prior='gauss', gen_lr=0.001,
    #                reg_lr=0.001, **ae_params),
    # DecodingRecommender(n_epochs=100, batch_size=100, optimizer='adam',
    #                     n_hidden=100, embedding=VECTORS,
    #                     lr=0.001, verbose=True)  # Only Title
    # VAERecommender(conditions=CONDITIONS, **vae_params),
    # DAERecommender(conditions=CONDITIONS, **ae_params)
    # Put more here...
]


def prepare_evaluation(bags, test_size=0.1, n_items=None, min_count=None):
    """
    Split data into train and dev set.
    Build vocab on train set and applies it to both train and test set.
    """
    # Split 10% validation data, one submission per day is too much.
    train_set, dev_set = bags.train_test_split(test_size=test_size)
    # Builds vocabulary only on training set
    # Limit of most frequent 50000 distinct items is for testing purposes
    vocab, __counts = train_set.build_vocab(max_features=n_items,
                                            min_count=min_count,
                                            apply=False)

    # Apply vocab (turn track ids into indices)
    train_set = train_set.apply_vocab(vocab)
    # Discard unknown tokens in the test set
    dev_set = dev_set.apply_vocab(vocab)

    # Drop one track off each playlist within test set
    noisy, missing = corrupt_sets(dev_set.data, drop=1)
    assert len(noisy) == len(missing) == len(dev_set)
    # Replace test data with corrupted data
    dev_set.data = noisy

    return train_set, dev_set, missing


def log(*print_args, logfile=None):
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        with open(logfile, 'a') as fhandle:
            print(*print_args, file=fhandle)
    print(*print_args)


def main(outfile=None, min_count=None):
    """ Main function for training and evaluating AAE methods on Reuters data """
    print("Loading data from", DATA_PATH)
    bags = Bags.load_tabcomma_format(DATA_PATH, unique=True)
    if args.compute_mi:
        from aaerec.utils import compute_mutual_info
        print("[MI] Dataset: Reuters")
        print("[MI] min Count:", min_count)
        tmp = bags.build_vocab(min_count=min_count, max_features=None)
        compute_mutual_info(tmp, conditions=None, include_labels=True,
                            normalize=True)
        print("=" * 78)
        exit(0)
    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)
    train_set, dev_set, y_test = prepare_evaluation(bags,
                                                    min_count=min_count)

    log("Train set:", logfile=outfile)
    log(train_set, logfile=outfile)

    log("Dev set:", logfile=outfile)
    log(dev_set, logfile=outfile)

    # THE GOLD (put into sparse matrix)
    y_test = lists2sparse(y_test, dev_set.size(1)).tocsr(copy=False)

    # the known items in the test set, just to not recompute
    x_test = lists2sparse(dev_set.data, dev_set.size(1)).tocsr(copy=False)

    for model in MODELS:
        log('=' * 78, logfile=outfile)
        log(model, logfile=outfile)

        # Training
        model.train(train_set)

        # Prediction
        y_pred = model.predict(dev_set)

        # Sanity-fix #1, make sparse stuff dense, expect array
        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()
        else:
            y_pred = np.asarray(y_pred)

        # Sanity-fix, remove predictions for already present items
        y_pred = remove_non_missing(y_pred, x_test, copy=False)

        # Evaluate metrics
        results = evaluate(y_test, y_pred, METRICS)

        log("-" * 78, logfile=outfile)
        for metric, stats in zip(METRICS, results):
            log("* {}: {} ({})".format(metric, *stats), logfile=outfile)

        log('=' * 78, logfile=outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.")
    parser.add_argument('-m', '--min-count', type=int,
                        default=None,
                        help="Minimum count of items")
    parser.add_argument('--compute-mi', default=False,
                        action='store_true')
    args = parser.parse_args()
    print(args)
    main(outfile=args.outfile, min_count=args.min_count)
