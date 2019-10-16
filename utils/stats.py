import collections
import os
import matplotlib

matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from aaerec.datasets import Bags
from eval.aminer import unpack_papers, papers_from_files
from eval.fiv import load as load_fiv, unpack_papers as unpack_papers_fiv
from eval.econis import load as load_econis, unpack_papers_conditions as unpack_papers_econis


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

# path = '/data21/lgalke/datasets/econbiz62k.tsv'
# path = '/data21/lgalke/datasets/PMC/citations_pmc.tsv'
# path = '/data22/ivagliano/Reuters/rcv1.tsv'
path = '/data22/ggerstenkorn/citation_data_preprocessing/final_data/owner_list_cleaned.csv'
dataset = "pubmed"

if dataset == "dblp" or dataset == "acm" or dataset == "swp" or dataset == "econis":
    if dataset == "dblp" or dataset == "acm":
        path = '/data22/ivagliano/aminer/'
        path += ("dblp-ref/" if dataset == "dblp" else "acm.txt")
        papers = papers_from_files(path, dataset, n_jobs=1)
    elif dataset == "econis":
        print("Loading Econis dataset")
        papers = load_econis("/data22/ivagliano/econis/econbiz62k-extended.json")
    else:
        print("Loading SWP dataset")
        papers = load_fiv("/data22/ivagliano/SWP/FivMetadata_clean.json")

    years, citations = {}, {}
    for paper in papers:
        if dataset != "econis":
            try:
                 years[paper["year"]] += 1
            except KeyError:
                if "year" not in paper.keys():
                    # skip papers without a year
                    continue
                years[paper["year"]] = 0
        else:
            try:
                years[paper["date"]] += 1
            except KeyError:
                if "date" not in paper.keys():
                    # skip papers without a year
                    continue
                years[paper["date"]] = 0
        if dataset == "dblp":
            try:
                citations[paper["n_citation"]] += 1
            except KeyError:
                citations[paper["n_citation"]] = 1
        elif dataset == "acm":
            if "references" not in paper.keys():
                continue
            for ref in paper["references"]:
                try:
                    citations[ref] += 1
                except KeyError:
                    citations[ref] = 1
        else:
            if "subjects" not in paper.keys():
                continue
            for subject in paper["subjects"]:
                try:
                    citations[subject] += 1
                except KeyError:
                    citations[subject] = 1

    years = collections.OrderedDict(sorted(years.items()))
    l = list(years.keys())
    print("First year {}, last year {}".format(l[0], l[-1]))
    cnt = 0

    for key, value in years.items():
        cnt += value
        if cnt/len(papers) >= 0.9:
            print("90:10 ratio at year {}".format(key))
            break

    print("Plotting paper distribution by year on file")
    # plot papers from 1970
    plot(years, dataset, "year", 1970)
    if dataset == "acm" or dataset == "swp" or dataset == "econis":
        citations = paper_by_n_citations(citations)

    citations = collections.OrderedDict(sorted(citations.items()))
    x_dim = "citations" if dataset != "swp" else "occurrences"
    print("Plotting paper distribution by number of {} on file".format(x_dim))
    # plot papers with at least 100 citations
    plot(citations, dataset, "number of {}".format(x_dim), 100)

    print("Unpacking {} data...".format(dataset))
    if dataset == "dblp" or dataset == "acm":
        bags_of_papers, ids, side_info = unpack_papers(papers)
    elif dataset == "swp":
        bags_of_papers, ids, side_info = unpack_papers_fiv(papers)
    else:
        bags_of_papers, ids, side_info = unpack_papers_econis(papers)
    bags = Bags(bags_of_papers, ids, side_info)

else:
    if dataset == "pubmed":
        print("Unpacking {} data...".format(dataset))
        # Only with more metadata (generic conditions) for Pubmed (Econis thorugh separate script /eval/econis.py)
        # key: name of a table
        # owner_id: ID of citing paper
        # fields: list of column names in table
        # target names: key for these data in the owner_attributes dictionary
        # path: absolute path to the csv file
        mtdt_dic =  collections.OrderedDict()
        mtdt_dic["author"] = {"owner_id": "pmId", "fields": ["name"],"target_names": ["author"],
                              "path": os.path.join("/data22/ggerstenkorn/citation_data_preprocessing/final_data/", "author.csv")}
        mtdt_dic["mesh"] = {"owner_id": "document", "fields": ["descriptor"], "target_names": ["mesh"],
                            "path": os.path.join("/data22/ggerstenkorn/citation_data_preprocessing/final_data/", "mesh.csv")}

        # With no metadata or just titles
        # bags = Bags.load_tabcomma_format(path, unique=True)
        # With more metadata for PubMed (generic conditions)
        bags = Bags.load_tabcomma_format(path, unique=True, owner_str="pmId",
                                         set_str="cited", meta_data_dic=mtdt_dic)
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
