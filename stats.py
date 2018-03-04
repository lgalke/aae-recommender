import numpy as np
from datasets import Bags
# path = '../Data/Economics/econbiz62k.tsv'
path = '../Data/PMC/citations_pmc.tsv'
bags = Bags.load_tabcomma_format(path, unique=True)
bags = bags.build_vocab(apply=True)

csr = bags.tocsr()
print("N ratings:", csr.sum())

column_sums = csr.sum(0).flatten()
row_sums = csr.sum(1).flatten()

print(column_sums.shape)
print(row_sums.shape)


FMT = "N={}, Min={}, Max={} Median={}, Mean={}, Std={}"

def compute_stats(A):
    return A.shape[1], A.min(), A.max(), np.median(A, axis=1)[0,0], A.mean(), A.std()


print("Items per document")
print(FMT.format(*compute_stats(row_sums)))
print("Documents per item")
print(FMT.format(*compute_stats(column_sums)))

