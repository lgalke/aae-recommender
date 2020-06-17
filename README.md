# aae-recommender

[![Build Status](https://travis-ci.org/lgalke/aae-recommender.svg?branch=master)](https://travis-ci.org/lgalke/aae-recommender)
[![DOI](https://zenodo.org/badge/DOI/10.1145/3267471.3267476.svg)](https://doi.org/10.1145/3267471.3267476)
[![DOI](https://zenodo.org/badge/DOI/10.1145/3209219.3209236.svg)](https://doi.org/10.1145/3209219.3209236)

Adversarial Autoencoders for Recommendation Tasks

## Dependencies

- torch
- numpy
- scipy
- sklearn
- gensim
- pandas
- joblib

If possible, numpy and scipy should be installed as system packages.
The dependencies `gensim` and `sklearn` can be installed via `pip`.
For pytorch, please refer to their [installation
instructions](http://pytorch.org/) that depend on the python/CUDA setup you are
working in.

To use pretreined word-embeddings, the [`word2vec` Google News](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) corpus should be download.

## Installation

You can install this package and all necessary dependencies via pip.

```sh
pip install -e .
```

## Running

The `main.py` file is an executable to run an evaluation of the specified models on the PubMed or `EconBiz` dataset (see the *Concrete datasets* section below).
The `dataset` and `year` are mandatory arguments. The `dataset` is expected to be a path to a tsv-file,
of which the format is described next.

The `eval/aminer.py` file is an executable to run an evaluation of the specified models on the AMiner datasets (see the *Concrete datasets* section below). The `dataset` and `year` are mandatory arguments. The `dataset` is expected to be either `dblp` or `acm`, and the `DATA_PATH` constant in the script needs to be set to the path to a folder which contains both datasets.

The `eval/rcv.py` file is an executable to run an evaluation of the specified models on the Reuters RCV1 dataset (see the *Concrete datasets* section below). The `DATA_PATH` constant in the script needs to be set to the path to a tsv-file,
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

The format of the ACM and DBLP datasets is described in their [AMiner](https://www.aminer.org/citation) documentation.

## Concrete datasets

We worked with the PubMed citations dataset from
[CITREC](https://www.isg.uni-konstanz.de/projects/citrec/).  We converted the
provided SQL dumps into the dataset format above.
The references in the CITREC TREC Genomics dataset are not disambiguated.
Therefore we operate only the PubMed dataset for citation recommendation.

For subject label recommendation, we used the the economics dataset `EconBiz`, provided by [ZBW](https://zbw.eu).

The PubMed and `EconBiz` datasets are available [here](https://www.kaggle.com/hsrobo/titlebased-semantic-subject-indexing).
For `EconBiz`, only titles are available and we are currently asserting that copyright issues do not prevent us from publishing the further metadata of the documents that we have used.

Further public datasets used were the DBLP-Citation-network V10 and ACM-Citation-network V9 datasets from the [AMiner](https://www.aminer.org/citation) project, and the [Reuters RCV1](https://trec.nist.gov/data/reuters/reuters.html) corpora.
We converted the provided XML dumps into the dataset format above.

We also run experiments with the Million Playlist Dataset (MPD), provided by [Spotify](https://research.spotify.com/datasets), and IREON, provided by [FIV](https://fiviblk.de/), but we are not allowed to redistribute them. The MPD dataset was used only to participate to the [RecSys Challenge 2018](http://www.recsyschallenge.com/2018/).

## References and cite

Please see our papers for additional information on the models implemented and the experiments conducted:

- [Multi-Modal Adversarial Autoencoders for Recommendations of Citations and Subject Labels](https://zenodo.org/record/1313577)

- [Using Adversarial Autoencoders for Multi-Modal Automatic Playlist Continuation](https://zenodo.org/record/1455214) 


If you use our code in your own work please cite one of these papers:

    @inproceedings{Vagliano:2018,
         author = {Vagliano, Iacopo and Galke, Lukas and Mai, Florian and Scherp, Ansgar},
         title = {Using Adversarial Autoencoders for Multi-Modal Automatic Playlist Continuation},
         booktitle = {Proceedings of the ACM Recommender Systems Challenge 2018},
         series = {RecSys Challenge '18},
         year = {2018},
         isbn = {978-1-4503-6586-4},
         location = {Vancouver, BC, Canada},
         pages = {5:1--5:6},
         articleno = {5},
         numpages = {6},
         url = {http://doi.acm.org/10.1145/3267471.3267476},
         doi = {10.1145/3267471.3267476},
         acmid = {3267476},
         publisher = {ACM},
         address = {New York, NY, USA},
         keywords = {adversarial autoencoders, automatic playlist continuation, multi-modal recommender, music recommender systems, neural networks},
    }

    @inproceedings{Galke:2018,
         author = {Galke, Lukas and Mai, Florian and Vagliano, Iacopo and Scherp, Ansgar},
         title = {Multi-Modal Adversarial Autoencoders for Recommendations of Citations and Subject Labels},
         booktitle = {Proceedings of the 26th Conference on User Modeling, Adaptation and Personalization},
         series = {UMAP '18},
         year = {2018},
         isbn = {978-1-4503-5589-6},
         location = {Singapore, Singapore},
         pages = {197--205},
         numpages = {9},
         url = {http://doi.acm.org/10.1145/3209219.3209236},
         doi = {10.1145/3209219.3209236},
         acmid = {3209236},
         publisher = {ACM},
         address = {New York, NY, USA},
         keywords = {adversarial autoencoders, multi-modal, neural networks, recommender systems, sparsity},
    }
  
