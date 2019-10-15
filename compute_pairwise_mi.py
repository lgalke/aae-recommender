"""
Executable to compute the Mutual Information across items variables

Theory: https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
Impl Docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
"""
import argparse
import numpy as np

from aaerec.datasets import Bags
from sklearn.metrics import mutual_info_score

from aaerec.condition import ConditionList, CountCondition
from aaerec.utils import compute_mutual_info

PARSER = argparse.ArgumentParser()
PARSER.add_argument('dataset', type=str,
                    help='path to dataset')
PARSER.add_argument('-m', '--min-count', type=int,
                    help='Pruning parameter', default=None)
PARSER.add_argument('-M', '--max-features', type=int,
                    help='Max features', default=None)
ARGS = PARSER.parse_args()


MI_CONDITIONS = ConditionList([('title', CountCondition(max_features=100000))])
# MI_CONDITIONS = None

print("Computing Mutual Info with args")
print(ARGS)

# With no metadata or just titles
BAGS = Bags.load_tabcomma_format(ARGS.dataset, unique=True)\
    .build_vocab(min_count=ARGS.min_count, max_features=ARGS.max_features)

compute_mutual_info(BAGS, MI_CONDITIONS, include_labels=True)
