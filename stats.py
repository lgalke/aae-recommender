import collections
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from aaerec.datasets import Bags
from aminer import unpack_papers, papers_from_files


def compute_stats(A):
    return A.shape[1], A.min(), A.max(), np.median(A, axis=1)[0,0], A.mean(), A.std()


def plot(objects, dataset, title, min_key):
    if min_key != -1:
    	objects = {x : objects[x] for x in objects if x >= min_key}
    y_pos = np.arange(len(objects.keys()))
    plt.bar(y_pos, objects.values(), align='center', alpha=0.5)
    plt.xticks(y_pos, objects.keys(), rotation='vertical')
    plt.ylabel('Papers')
    plt.title('Papers by {}'.format(title))
    plt.savefig('papers_by_{}_{}.pdf'.format(title.replace(" ", "_"), dataset))
    # plt.show()
    plt.close()


def paper_by_n_citations(citations):
    '''
    From a dictionary with paper IDs as keys and citation numbers as values
    to a dictionary with citation numbers as keys and paper numbers as values
    '''
    papers_by_citations = {}
    for paper in citations.keys():
        try:
            papers_by_citations[citations[paper]] += 1
        except KeyError:
            papers_by_citations[citations[paper]] = 1

    return papers_by_citations

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
            if "year" not in paper.keys():
                # skip papers without a year
                continue
            years[paper["year"]] = 0
        if dataset == "dblp":
            try:
                citations[paper["n_citation"]] += 1
            except KeyError:
                citations[paper["n_citation"]] = 1
        else:
            if "references" not in paper.keys():
                continue
            for ref in paper["references"]:
                try:
                    citations[ref] += 1
                except KeyError:
                    citations[ref] = 1

    years = collections.OrderedDict(sorted(years.items()))
    sum = 0

    for key, value in years.items():
        sum += value
        if sum/len(papers) >= 0.9:
            print("90:10 ratio at year {}".format(key))
            break

    print("Plotting paper distribution by year on file")
    # plot papers from 1970
    plot(years, dataset, "year", 1970)
    if dataset == "acm":
        citations = paper_by_n_citations(citations)

    citations = collections.OrderedDict(sorted(citations.items()))
    print("Plotting paper distribution by number of citations on file")
    #plot papers with at least 100 citations
    plot(citations, dataset, "number of citations", 100)

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
