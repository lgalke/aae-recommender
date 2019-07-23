import collections

import matplotlib

matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from aaerec.datasets import Bags
from eval.aminer import unpack_papers, papers_from_files
from eval.fiv import load, unpack_papers as unpack_papers_fiv
from eval.mpd.mpd import playlists_from_slices, unpack_playlists
import pandas as pd


def compute_stats(A):
    return A.shape[1], A.min(), A.max(), np.median(A, axis=1)[0,0], A.mean(), A.std()


def plot(objects, dataset, title, x=None):
    # y_pos = np.arange(len(objects.keys()))
    plt.bar(objects.keys(), objects.values(), align='center', alpha=0.5)
    # plt.xticks(y_pos, objects.keys(), rotation='vertical')
    if dataset == "mpd":
        label = "Tracks"
    elif dataset == "swp" or dataset == "rcv" or dataset == "econbiz":
        label = 'Labels'
    else:
        label = 'Papers'
    plt.ylabel(label)
    # plt.title('Papers by {}'.format(title))
    plt.xlabel(title)

    # print the y value of bar at a given x
    if x != None:
        for x_i, y_i in enumerate(objects.values()):
            if x_i == x:
                print("For x={} y={}".format(x, y_i))
                plt.text(x_i, y_i, str(y_i) + "\n", ha='center')

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


def from_to_key(objects, min_key, max_key=None):
    '''
    From a dictionary create a new dictionary with only items with key greater
    than min_key and optionally smaller than max_key. If min_key = max_key,
    it returns a dictionaly with only that key
    '''
    print("Filtering dictionary's keys from {} to {}".format(min_key, max_key))
    # It assumes only positive integers as keys
    if max_key != None and max_key < min_key:
        print("Error: max_key has to be greater than min_key. Dictionary unchanged")
        return objects

    if  max_key == None and min_key <= 0:
        print("Warning: min_key lower or equal to 0 and no max_key has no effect."
            + "Dictionary unchanged")
        return objects

    if max_key != None:
        return {x : objects[x] for x in objects if x >= min_key and x <= max_key}

    return {x : objects[x] for x in objects if x >= min_key}


def generate_years_citations(papers, dataset):
    '''
    Return the distribution of papers by years and by citations
    '''
    years, citations = {}, {}

    for paper in papers:
        try:
             years[paper["year"]] += 1
        except KeyError:
            # MPD has no time information (no year)
            if "year" not in paper.keys() and dataset != "mpd":
                # skip papers without a year
                # unless dataset is MPD, which has no year
                continue
            if dataset != "mpd":
                years[paper["year"]] = 0
        if dataset == "dblp":
            # DBLP has the citations for each paper
            try:
                citations[paper["n_citation"]] += 1
            except KeyError:
                citations[paper["n_citation"]] = 1
        elif dataset == "acm":
            # For ACM we need to compute the citations for each paper
            if "references" not in paper.keys():
                continue
            for ref in paper["references"]:
                try:
                    citations[ref] += 1
                except KeyError:
                    citations[ref] = 1
        elif dataset == "swp":
            # For SWP we need to compute the occurrences for each subject
            if "subjects" not in paper.keys():
                continue
            for subject in paper["subjects"]:
                try:
                    citations[subject] += 1
                except KeyError:
                    citations[subject] = 1
        else:
            # For MPD we need to compute the occurrences for each track
            for track in paper["tracks"]:
                try:
                    citations[track["track_uri"]] += 1
                except KeyError:
                    citations[track["track_uri"]] = 1

    return years, citations


def generate_citations(df):
    citations = {}

    for index, paper in df.iterrows():
        for ref in paper["set"].split(","):
            if ref == "":
                continue
            try:
                citations[ref] += 1
            except KeyError:
                citations[ref] = 1

    return citations


def set_count(df):
    set_cnts = {}

    for index, paper in df.iterrows():
        set_cnts[paper["owner"]] = len(paper["set"].split(","))

    return set_cnts


def set_path(ds):

    if ds == "dblp" or ds == "acm":
        p = '/data22/ivagliano/aminer/'
        p += ("dblp-ref/" if ds == "dblp" else "acm.txt")
    elif ds == "swp":
        p = "/data22/ivagliano/SWP/FivMetadata_clean.json"
    elif ds == "mpd":
        p = "/data21/lgalke/datasets/MPD/data/"
    elif ds == "pubmed":
        p = "/data21/lgalke/datasets/PMC/citations_pmc.tsv"
    elif ds == "econbiz":
        p = "/data21/lgalke/datasets/econbiz62k.tsv"
    else:
        p = "/data22/ivagliano/Reuters/rcv1.tsv"

    return p


# path = '/data21/lgalke/datasets/econbiz62k.tsv'
# path = '/data21/lgalke/datasets/PMC/citations_pmc.tsv'
# path = '/data22/ivagliano/Reuters/rcv1.tsv'

# Possible values: pubmed, dblp, acm, swp, rcv, econbiz, mpd
dataset = "pubmed"
# only papers/labels with at least min_x_cit citations/occurrences
# in the plot of the distribution of papers/labels by citations/occurrences
# Set to 0 if not relevant
min_x_cit = 10
# only papers/labels with at most man_x_cit citations/occurrences
# in the plot of the distribution of papers/labels by citations/occurrences
# Set to None if not relevant
max_x_cit = 100
# Shows the y-value at the given mark_x_cit
# Set to None if not relevant
mark_x_cit = 50

path = set_path(dataset)

if dataset == "dblp" or dataset == "acm" or dataset == "swp" or dataset == "mpd":
    if dataset != "swp" and dataset != "mpd":
        # path = '/data22/ivagliano/aminer/'
        # path += ("dblp-ref/" if dataset == "dblp" else "acm.txt")
        print("Loading {} dataset".format(dataset.upper()))
        papers = papers_from_files(path, dataset, n_jobs=1)
    elif dataset == "swp":
        print("Loading SWP dataset")
        papers = load(path)
    else:
        print("Loading MPD dataset")
        # actually not papers but playlists
        papers = playlists_from_slices(path, n_jobs=4)

    years, citations = generate_years_citations(papers, dataset)

    if dataset != "mpd":
        # only papers from 1970 
        years = from_to_key(years, 1970)
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
        plot(years, dataset, "year")

    if dataset == "acm" or dataset == "swp" or dataset == "mpd":
        if dataset == "acm":
            text = "citations"
        elif dataset == "swp":
            text = "labels"
        else:
            text = "tracks"
        print("Generating {} distribution".format(text))
        citations = paper_by_n_citations(citations)

    # only papers with at least 100 citations
    # citations = from_to_key(citations, 100)
    # only papers with min min_x_cit citations and max_x_cit citations
    citations = from_to_key(citations, min_x_cit, max_x_cit)
    citations = collections.OrderedDict(sorted(citations.items()))
    x_dim = "Citations" if dataset != "swp" and dataset != "mpd" else "Occurrences"

    print("Plotting paper distribution by number of {} on file".format(x_dim.lower()))
    # show the y-value for the bar at x=mark_x_cit in the plot
    plot(citations, dataset, x_dim, mark_x_cit)
    # show no y-value for any bar
    # plot(citations, dataset, x_dim)

    print("Unpacking {} data...".format(dataset))
    if dataset == "acm" or dataset == "dblp":
        bags_of_papers, ids, side_info = unpack_papers(papers)
    elif dataset == "mpd":
        # not bags_of_papers but bugs_of_tracks
        bags_of_papers, ids, side_info = unpack_playlists(papers)
    else:
        bags_of_papers, ids, side_info = unpack_papers_fiv(papers)
    bags = Bags(bags_of_papers, ids, side_info)

else:
    print("Loading {}".format(path))
    df = pd.read_csv(path, sep="\t", dtype=str, error_bad_lines=False)
    # replace nan with empty string
    df = df.replace(np.nan, "", regex=True)

    citations = generate_citations(df)
    citations = paper_by_n_citations(citations)
    # only papers with at least 10 citations
    # citations = from_to_key(citations, 10)
    # only papers with min min_x_cit and max max_x_cit citations
    citations = from_to_key(citations, min_x_cit, max_x_cit)
    citations = collections.OrderedDict(sorted(citations.items()))
    x_dim = "Citations" if dataset == "pubmed" else "Occurrences"

    print("Plotting {} distribution by number of {} on file"
          .format("papers'" if x_dim == "Citations" else "labels'", x_dim.lower()))
    # show the y-value for the bar at x=mark_x_cit in the plot
    plot(citations, dataset, x_dim, mark_x_cit)
    # show no y-value for any bar
    # plot(citations, dataset, x_dim)

    set_cnts = set_count(df)
    x_dim = "References" if dataset == "pubmed" else "Labels"
    print("Plotting papers' distribution by number of their {} on file".format(x_dim.lower()))
    # show the y-value for the bar at x=50 in the plot
    # plot(citations, dataset, x_dim, 100)
    # show no y-value for any bar
    plot(citations, dataset, x_dim)

    print("Unpacking {} data...".format(dataset))
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
