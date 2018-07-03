import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from aaerec.datasets import Bags
from aminer import unpack_papers, papers_from_files


def compute_stats(A):
    return A.shape[1], A.min(), A.max(), np.median(A, axis=1)[0,0], A.mean(), A.std()


def plot(objects, dataset, title):
    y_pos = np.arange(len(objects.keys()))
    plt.bar(y_pos, objects.values(), align='center', alpha=0.5)
    plt.xticks(y_pos, objects.keys())
    plt.ylabel('Papers')
    plt.title('Papers by {}'.format(title))
    plt.savefig('papers_by{}_{}.pdf'.format(title, dataset))


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
        if dataset == "dblp":
            try:
                citations[paper["n_citation"]] += 1
            except KeyError:
                citations = [paper["n_citation"]] = 0

    plot(years, dataset, "year")
    if dataset == "dblp":
        plot(citations, dataset, "number of citations")

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

print("Items per document")
print(FMT.format(*compute_stats(row_sums)))
print("Documents per item")
print(FMT.format(*compute_stats(column_sums)))

