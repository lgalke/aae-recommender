# aae-recommender

Adversarial Autoencoders for Recommendation Tasks

## Dependencies

- torch
- numpy
- scipy
- sklearn
- gensim

If possible, numpy and scipy should be installed as system packages.
The dependencies `gensim` and `sklearn` can be installed via `pip`.
For pytorch, please refer to their [installation
instructions](http://pytorch.org/) that depend on the python/CUDA setup you are
working in.

## Running

The `main.py` file is an executable to run an evaluation of the specified models.
The `dataset` and `year` are mandatory arguments. The `dataset` is expected to be a path to a tsv-file,
of which the format is described next.

## Dataset Format


The expected dataset Format is a **tab-separated** with columns:

- **owner** id of the document
- **set** comma separated list of items
- **year** year of the document
- **title** of the document

The columns 'owner' and 'set' are expected to be the first two ones, since they are mandatory.
An arbitrary number of supplementary information columns can follow.
The current implementation, however, makes use of the `year` property for splitting the data into train and test sets.
Also, title-enhanced recommendation models rely on the `title` property to be present.

## Concrete datasets

So far, we worked with the PubMed citations dataset from
[CITREC](https://www.isg.uni-konstanz.de/projects/citrec/).  We converted the
provided SQL dumps into the dataset format above. We plan to also publish our
converted tsv version of the CITREC PubMed dataset.
In the CITREC TREC Genomics dataset, there references are not disambiguated.
Therefore we operate only the PubMed dataset for citation recommendation.

For subject label recommendation, we used the the economics dataset `EconBiz`, provided by [ZBW](https://zbw.eu).
We are currently asserting that copyright issues do not prevent us from publishing the
metadata of the documents in the aforementioned format.
