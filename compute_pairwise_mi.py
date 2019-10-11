"""
Executable to compute the Mutual Information across items variables

Theory: https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
Impl Docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
"""
import argparse
import numpy as np

from aaerec.datasets import Bags
from sklearn.metrics import mutual_info_score

PARSER = argparse.ArgumentParser()
PARSER.add_argument('dataset', type=str,
                    help='path to dataset')
PARSER.add_argument('-m', '--min-count', type=int,
                    help='Pruning parameter', default=None)
PARSER.add_argument('-M', '--max-features', type=int,
                    help='Max features', default=None)
ARGS = PARSER.parse_args()


print("Computing Mutual Info with args")
print(ARGS)

# With no metadata or just titles
X = Bags.load_tabcomma_format(ARGS.dataset, unique=True)\
    .build_vocab(min_count=ARGS.min_count, max_features=ARGS.max_features)\
    .tocsr()

# Co-occurrence matrix, as in `Countbased`
C = X.T @ X
print("(Pairwise) Mutual information:", mutual_info_score(None, None, contingency=C))
