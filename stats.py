import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from aaerec.datasets import Bags
from aminer import unpack_papers, papers_from_files
# path = '../Data/Economics/econbiz62k.tsv'
path = '/data21/lgalke/PMC/citations_pmc.tsv'
dataset = "dblp"

if dataset == "dblp" or dataset == "acm":
    path = '/data22/ivagliano/aminer/'
    path += ("dblp-ref/" if dataset == "dblp" else "acm.txt")
    papers = papers_from_files(path, dataset, n_jobs=1)

    years, citations = {}, {}
    for paper in papers:
        try:
            years[paper["year"]] += 1
        except KeyError:
            years[paper["year"]] = 0
        try:
            citations[paper["citations"]] += 1
        except KeyError:
            citations = [paper["citations"]] = 0

    y_pos = np.arange(len(years.keys()))
    plt.bar(y_pos, years.values(), align='center', alpha=0.5)
    plt.xticks(y_pos, years.keys())
    plt.ylabel('Papers')
    plt.title('Papers per year')
    plt.savefig('papersPerYear_{}.pdf'.format(dataset))
    
    y_pos = np.arange(len(citations.keys()))
    plt.bar(y_pos, citations.values(), align='center', alpha=0.5)
    plt.xticks(y_pos, citations.keys())
    plt.ylabel('Papers')
    plt.title('Papers by number of citations')
    plt.savefig('papersByCitations_{}.pdf'.format(dataset))

    print("Unpacking {} data...".format(dataset))
    bags_of_papers, ids, side_info = unpack_papers(papers)
    bags = Bags(bags_of_papers, ids, side_info)

else:
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

