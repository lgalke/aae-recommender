# coding: utf-8
import math
import random
from builtins import filter # nanu?
from collections import Counter
import itertools as it

import numpy as np
import pandas as pd

from tqdm import tqdm


from sklearn.model_selection import train_test_split
from collections import defaultdict
try:
    from .transforms import lists2sparse, sparse2lists
except SystemError:
    from transforms import lists2sparse, sparse2lists

def split_by_mask(data, condition):
    """ Splits data on index depending on condition """
    truthy = [d for i, d in enumerate(data) if condition[i]]
    falsy = [d for i, d in enumerate(data) if not condition[i]]
    return truthy, falsy



def magic(S, N, alpha=0.05):
    return S**2 * math.log(S * N / alpha)


def build_vocab(sets, min_count=None, max_features=None):
    """ Builds the vocabulary with respect to :code:`max_features` most common
    tokens and :code:`min_count` minimal set length """
    # sort descending by frequency and only keep max_features
    counts = Counter(it.chain.from_iterable(sets)).most_common(max_features)
    # [ ('token', 42) ]

    # can be optimized
    if min_count:
        counts = list(it.takewhile(lambda c: c[1] >= min_count, counts))
        # counts = list(filter(lambda c: c[1] >= min_count, counts))

    # incrementally assign indicies to tokens
    vocab = {}

    for token, __ in counts:
        vocab[token] = len(vocab)

    return vocab, counts


def filter_vocab(lists, vocab):
    """
    Filters out-of-vocabulary tokens from iterable of iterables.
    We could impose special UNK treatment here.
    """
    return [[t for t in tokens if t in vocab] for tokens in lists]


def apply_vocab(lists, vocab):
    """ Applys vocab to iterable of iterables. This function can also be used
    to invert the mapping. """
    return [[vocab[t] for t in l] for l in lists]

def filter_apply_vocab(lists, vocab):
    """ Filters for valid tokens and transforms them to index. Faster then
    first filter then apply. """
    return [[vocab[t] for t in tokens if t in vocab] for tokens in lists]

def filter_length(lists, min_length, *supplements):
    """
    Filters lists and supplements with respect to len(list) > min_length
    """
    enough = [len(bag) >= min_length for bag in lists]
    lists_reduced = [bag for i, bag in enumerate(lists) if enough[i]]
    if not supplements:
        return lists_reduced

    sup_reduced = []
    for supplement in supplements:
        sup_reduced.append([o for i, o in enumerate(supplement) if enough[i]])
    return (lists_reduced, *sup_reduced)



def split_set(s, criterion):
    """
    Splits a set according to criterion
    if criterion is float: toss a coin for each element
    if criterion is an int: drop as many random elements
    if criterion is callable: drop each element iff criterion(element) returns
    False

    In either case, the result is (remainder_set, dropped_elements)
    """
    s = set(s)

    if callable(criterion):
        todrop = {e for e in s if criterion(e)}
    elif type(criterion) == float:
        assert criterion > 0 and criterion < 1, "Float not bounded in (0,1)"
        todrop = {e for e in s if random.random() < criterion}
    elif type(criterion) == int:
        try:
            todrop = random.sample(s, criterion)
        except ValueError:  # too few elements in s
            todrop = s
    else:
        raise ValueError('int, float, or callable expected')

    todrop = set(todrop)
    return s - todrop, todrop


def corrupt_sets(sets, drop=1):
    """

    Splits a list of sets into two sub-sets each,
    one containing corrupted sets and one retaining the removed elements
    """
    split = [split_set(s, drop) for s in sets]

    return tuple(zip(*split))


class Bags(object):
    """
    Loads the data set and creates the mapping from tokens to indices

    1. Split line by tab and extract the owner token from the first column and
       split the second column by comma to obtain sets
    2. Use remaining columns to extract attributes of the owner
    3. Count element occurrences of these sets and retain ``max_featues``
        most common.
    3. Assign indices to tokens (descending with respect to their count)
    4. Remove sets that then have less than ``min_elements`` elements.
    5. Store mappings and data
            * ``vocabulary`` : token -> index
            * ``index2token`` : index -> token
            * ``data`` : sets with tokens replaced by indices
            * ``owner_attributes`` : dictionary of dictionaries, where
              owner_attributes[<attribute>][<token>] returns the <attribute>
              value of owner <token>
            * (DEPRECATED) ``element_attributes`` : dictionary of dictionaries, where
              element_attributes[<attribute>][<token>] returns the <attribute>
              value of set element <token>

    There is no special treatment for ``'UNK'`` tokens.

    >>> bags = Bags.from_sets([[1,2,3] , [2,3,4,1] , [42, 1, 7]])
    >>> len(bags)
    3
    >>> bags.size()
    (3, 6)
    """
    def __init__(self,
                 data,
                 owners,
                 owner_attributes=None):
        assert len(owners) == len(data)
        self.data = data
        self.bag_owners = owners
        self.owner_attributes = owner_attributes

        # consumes performance
        # self.index2token = invert_mapping(vocab)


    def clone(self):
        """ Creates a really deep copy """
        data = [[t for t in b] for b in self.data]
        bag_owners = [o for o in self.bag_owners]
        if self.owner_attributes is not None:
            owner_attributes = {attr: {token: value for token, value in value_by_token.items()}
                                for attr, value_by_token in self.owner_attributes.items()}

        return Bags(data, bag_owners, owner_attributes=owner_attributes)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "{} records with {} ratings".format(len(self), self.numel())

    def __getitem__(self, idx):
        return self.data[idx]

    def maxlen(self):
        """ Returns the maximum bag length """
        return max(map(len, self.data))

    def numel(self):
        """ Computes the number of (non-zero) elements """
        return sum(map(len, self.data))



    def get_attribute(self, attribute):
        """
        Retrieves the attribute 'attribute' of each bag owner and returns them as a list
        in the same order as self.bag_owners.
        """
        if self.owner_attributes is None or self.bag_owners is None:
            raise ValueError("Owners not present")

        attributes = []
        for owner in self.bag_owners:
            attributes.append(self.owner_attributes[attribute][owner])

        return attributes

    ### Attribute features to be dropped ###

    # def reverse_vocab(self):
    #     # This essentially recomputes index2token
    #     reverse = {}
    #     for token in self.vocab:
    #         reverse[self.vocab[token]] = token
    #     return reverse

    # def get_element_attributes(self, attribute, default_value = None):
    #    """
    #    Returns a list of element attributes sorted in the same order as the index.
    #    Hence, bag.get_element_attributes("<attribute>")[i] returns the value of <attribute>
    #    of the element represented in column bag.tocsr()[:, i].

    #    Arguments:
    #    ==========
    #    attribute : str
    #        The name of the attribute.
    #    default_value : default = None
    #        If a token can not be found in the vocabulary, this value is inserted instead.
    #    """
    #    #reverse_voc = self.reverse_vocab()
    #    reverse_voc = self.index2token
    #    indices = [idx for idx in reverse_voc]  # No-op
    #    indices = sorted(indices)

    #    attribute_values = []
    #    for idx in indices:
    #        att_dict = self.element_attributes[attribute]
    #        token = reverse_voc[idx]
    #        if token in att_dict:
    #            attribute_values.append(att_dict[token])
    #        else:
    #            attribute_values.append(default_value)

    #    return attribute_values

    # def populate_element_attributes(self, element_attributes_file, attributes):
    #    """
    #    Fills the bag with element attributes by extracting them from a file.
    #    Iterates over all elements in all sets, extracts their token, and fills the element_attributes dictionary with
    #    respective values.

    #    Arguments:
    #    =========
    #    element_attributes_file : str
    #        A .CSV file that contains the values for 'attributes'. Must contain a column 'token' that contains
    #        the token of the element to retrieve the attribute for, and it must contain a column for each attribute field in the
    #        list of attributes <attributes>.
    #    attributes : list of str
    #        Contains the names of the attributes to extract from the file.
    #    """
    #    tokens_in_bag = set([key for key in self.vocab])
    #    df = pd.read_csv(element_attributes_file, dtype = {"token" : str})
    #    tokens_in_df = set(df["token"].values)

    #    tokens_in_df_and_bag = tokens_in_df.intersection(tokens_in_bag)
    #    relevant_df = df[df["token"].isin(tokens_in_df_and_bag)]

    #    for row in relevant_df.iterrows():
    #        row = row[1]
    #        for att_name in attributes:
    #            self.element_attributes[att_name][row["token"]] = row[att_name]

    ### Attribute features to be dropped ###

    # @classmethod
    # def from_sets(
    #         self,
    #         sets,
    #         set_owners,  # stores the identifier associated with each bag, i.e., the id of the citing document
    #         owner_attributes,
    #         max_features=None,
    #         min_count=None,  # min cited
    #         min_elements=1):  # min citing
    #     # this corrups natural order.
    #     print("Removing duplicate elements...")
    #     sets = [list(set(x)) for x in sets]  # remove duplicates
    #     print("Data:", len(sets))
    #     vocab, counts = build_vocab(
    #         sets, max_features=max_features, min_count=min_count)
    #     print("Pruned vocab has", len(vocab), "elements with minimum",
    #           min_count, "occurences.")
    #     # TMP disable Dont use
    #     # print("Filtering set owners...")
    #     # set_owners = filter_set_owners(set_owners, sets, vocab, min_elements)
    #     # set_owners = filter_set_owners_slow(set_owners, sets, max_features,
    #     #                                     min_count, min_elements)
    #     # ^^^ this is essentially the same vvv there must be a better way
    #     print("Filtering rare tokens...")
    #     sets = filter_vocab(sets, vocab)
    #     print("Converting tokens to indices...")
    #     data = apply_vocab(sets, vocab)
    #     # filter too small sets
    #     print("First 5 bags", data[:5])
    #     if min_elements:
    #         print("Filtering sets that contain minimum", min_elements,
    #               "elements")
    #         enough = [len(bag) >= min_elements for bag in data]
    #         data = [b for i, b in enumerate(data) if enough[i]] 
    #         set_owners = [o for i, o in enumerate(set_owners) if enough[i]] 
    #         # filtered = filter(lambda row: len(row[0]) >= min_elements,
    #         #                   zip(data, set_owners))
    #         # data, set_owners = tuple(zip(*filtered))
    #         # could also filter owner_attributes here...
    #     assert len(data) == len(set_owners)
    #     print("Retained", len(data), "records.")
    #     # TODO FIXME pass constraints to store in Bags class for rebuilding vocab
    #     bags = Bags(data, vocab, counts, set_owners, owner_attributes)

    #     return bags

    @classmethod
    def load_tabcomma_format(self, path, unique=False):
        """
        Arguments
        =========

        """
        # loading
        # TODO FIXME make the year and the month column int? c0
        df = pd.read_csv(path, sep="\t", dtype=str, error_bad_lines=False)
        df = df.fillna("")

        header = list(df.columns.values)
        owner_attributes = dict()
        sets = df["set"].values
        set_owners = df["owner"].values
        print("Found", len(sets), 'rows')

        meta_vals = []
        for meta_header in header[2:]:
            meta_vals.append(df[meta_header].values)
        print("with", len(header) - 2, "metadata columns.")


        # for i, owner in enumerate(set_owners):
        #     for j in range(2, len(header)):
        #         owner_attributes[header[j]][owner] = meta_vals[j - 2][i]

        for i in range(2, len(header)):
            owner_attributes[header[i]] = {}
            for j, owner in enumerate(set_owners):
                owner_attributes[header[i]][owner] = meta_vals[i - 2][j]

        sets = list(map(lambda x: x.split(","), sets))
        if unique:
            print("Making items unique within user.")
            sets = [list(set(s)) for s in sets]


        bags = Bags(sets, set_owners, owner_attributes=owner_attributes)

        return bags

    def train_test_split(self, on_year=None, **split_params):
        """ Returns one training bags instance and one test bag instance.
        Builds the vocabulary from the training set.
        """
        if on_year is not None:
            print("Splitting data on year:", on_year)
            assert self.owner_attributes['year'], "Cant split on non-existing 'year'"
            on_year = int(on_year)
            is_train = [int(y) < on_year for y in self.get_attribute('year')]
            train_data, test_data = split_by_mask(self.data, is_train)
            train_owners, test_owners = split_by_mask(self.bag_owners, is_train)
        else:
            print("Splitting on params:", split_params)
            split = train_test_split(self.data, self.bag_owners, **split_params)
            train_data, test_data, train_owners, test_owners = split
        print("{} train, {} test documents.".format(len(train_data), len(test_data)))
        metadata_columns = list(self.owner_attributes.keys())
        train_attributes = {k: {owner: self.owner_attributes[k][owner] for owner in
                                train_owners} for k in metadata_columns}
        test_attributes = {k: {owner: self.owner_attributes[k][owner] for owner in
                               test_owners} for k in metadata_columns}
        train_set = Bags(train_data, train_owners, owner_attributes=train_attributes)
        test_set = Bags(test_data, test_owners, owner_attributes=test_attributes)
        return train_set, test_set

    def build_vocab(self, min_count=None, max_features=None, apply=True):
        """
        Returns BagsWithVocab instance if apply is True, else vocabulary along with counts.
        """
        vocab, counts = build_vocab(self.data, min_count=min_count,
                                    max_features=max_features)
        if apply:
            return self.apply_vocab(vocab)

        return vocab, counts


    def apply_vocab(self, vocab):
        """
        Applies `vocab` and returns `BagsWithVocab` instance
        """
        data_ix = filter_apply_vocab(self.data, vocab)
        return BagsWithVocab(data_ix, vocab, owners=self.bag_owners,
                             attributes=self.owner_attributes)

    def prune_(self, min_elements=0):
        """ Prunes data and set_owners according such that only rows with
        min_elements in data are retained """
        data = self.data
        owners = self.bag_owners
        if min_elements:
            data, owners = filter_length(data, min_elements, owners)
            attributes = {k: {owner: self.owner_attributes[k][owner] for owner
                              in owners} for k in
                          list(self.owner_attributes.keys())}
        self.data = data
        self.bag_owners = owners
        self.owner_attributes = attributes
        return self

    # END BASIC FUNCTIONALITY

    # def split(self, is_test, is_train = lambda x : True):
    #     """
    #     Splits the bag into two parts between which the data is disjoint.

    #     Parameters:
    #         - is_test: A function 'token' -> Bool that determines for each
    #         token whether it should belong to the second set returned or not.
    #         - is_train: A function 'token' -> Bool that determines for each
    #         token whether it should belong to the first set returned or not.
    #         Note that it will only be assigned to the first set if it has not
    #         already been assigned to the second one. By default, it will
    #         automatically be assigned
    #         to the first set if is_test yields False for the token.

    #     Returns:
    #         - train_bag : The first bag
    #         - test_bag : The second bag
    #     """
    #     raise DeprecationWarning("Use train_test_split() instead.")

    #     # split owners
    #     train_owners = []
    #     test_owners = []

    #     n_ignored = 0
    #     for i, owner in enumerate(self.bag_owners):
    #         # are 2 *different* criterions really necessary TODO FIXME
    #         if is_test(owner):
    #             test_owners.append(i)
    #         elif is_train(owner):
    #             train_owners.append(i)
    #         else:
    #             n_ignored += 1
    #             # raise ValueError("{} is neither train nor test.".format(owner))

    #     print("{} train documents, {} test documents".format(len(train_owners), len(test_owners)))
    #     print(n_ignored, "documents are neither train nor test.")

    #     train_bag_owners = [self.bag_owners[i] for i in train_owners]
    #     test_bag_owners = [self.bag_owners[i] for i in test_owners]

    #     # select data columns accordingly
    #     train_data = [self.data[i] for i in train_owners]
    #     test_data = [self.data[i] for i in test_owners]

    #     # build new bags
    #     train_bag = Bags(
    #         train_data,
    #         self.vocab,
    #         self.counts,
    #         train_bag_owners,
    #         owner_attributes=self.owner_attributes)
    #     test_bag = Bags(
    #         test_data,
    #         self.vocab,
    #         self.counts,
    #         test_bag_owners,
    #         owner_attributes=self.owner_attributes)

    #     return train_bag, test_bag

    def inflate(self, factor):
        # TODO FIXME
        """
        Inflates the bag by injecting 'factor' repetitions of the current data (and respective owner) into the bag.

        Parameters:
        =============
        factor : int
            Determines how often the data is repeated in the bag.


        Returns:
        =============
        inflated_bag : The same Bags instance on which the function is called.
        """

        # no
        # current_data = self.data.copy()
        # current_owners = self.bag_owners.copy()

        for __i in range(1, factor):
            self.data.extend([[t for t in b] for b in self.data])
            self.bag_owners.extend([o for o in self.bag_owners])

        return self


class BagsWithVocab(Bags):
    def __init__(self, data, vocab, owners=None, attributes=None):
        super(BagsWithVocab, self).__init__(data, owners,
                                            owner_attributes=attributes)
        self.vocab = vocab
        # array of tokens which acts as reverse vocab
        self.index2token = {v: k for k, v in vocab.items()}

    def clone(self):
        """ Creates a really deep copy """
        # safe cloning of an instance
        # deepcopy is NOT enough
        data = [[t for t in b] for b in self.data]
        vocab = {k: v for k, v in self.vocab.items()}
        bag_owners = [o for o in self.bag_owners]
        if self.owner_attributes is not None:
            attributes = {attr: {token: value for token, value in value_by_token.items()}
                          for attr, value_by_token in self.owner_attributes.items()}

        return BagsWithVocab(data, vocab, owners=bag_owners,
                             attributes=attributes)

    def build_vocab(self, min_count=None, max_features=None, apply=True):
        """ Override to prevent errors like building vocab of indices """
        raise ValueError("Instance already has vocabulary.")

    def apply_vocab(self, vocab):
        """ Override to prevent errors like building vocab of indices """
        raise ValueError("A vocabulary has already been applied.")

    def __str__(self):
        s = "{} elements in [{}, {}] with density {}"
        return s.format(self.numel(), *self.size(), self.density())

    def size(self, dim=None):
        sizes = (len(self.data), len(self.vocab))

        if dim:
            return sizes[dim]
        else:
            return sizes

    def tocsr(self, data=None):
        """
        Use ``size`` to transform into scipy.sparse format
        """

        if data is None:
            data = self.data
            size = self.size()
        else:
            size = len(data), self.size(1)

        return lists2sparse(data, size).tocsr()

    def train_test_split(self, **split_params):
        """ Returns one training bags instance and one test bag instance.
        Builds the vocabulary from the training set.
        """
        train_bags, test_bags = super().train_test_split(**split_params)
        train_set = BagsWithVocab(train_bags.data, self.vocab, owners=train_bags.bag_owners,
                                  attributes=train_bags.owner_attributes)
        test_set = BagsWithVocab(test_bags.data, self.vocab, owners=test_bags.bag_owners,
                                 attributes=test_bags.owner_attributes)
        return train_set, test_set

    def density(self):
        """ Computes the density: number of elements divided by dimensions """
        return self.numel() / np.product(self.size())

    def magic_number(self, std_factor=None, alpha=0.05):
        """ Computes the magic number for sparse retrieval with given error
        probability :code:`alpha`. Optionally adds fractions of the standard deviation of set length """
        lens = np.array(list(map(len, self.data)))
        S = lens.mean()

        if std_factor:
            S += std_factor * lens.std()

        return int(magic(S, self.size(1), alpha=alpha)) + 1

    def to_index(self, data):
        """
        Uses 'vocabulary' to transforms a list of
         sets of tokens ('data') to a list of sets of indices.
        """
        raise DeprecationWarning("Use apply_vocab(data, self.vocab) instead")
        return apply_vocab(data, self.vocab)

    def to_tokens(self, data):
        """
        Uses 'index2token' to transforms a list of
         sets of indices ('data') to a list of sets of tokens.
        """
        raise DeprecationWarning("Use apply_vocab(data, self.index2token) instead")
        return apply_vocab(data, self.index2token)

    def raw(self):
        """ Returns the data with original identifiers instead of indices """
        # important not to filter here, better raise if something wrong
        return apply_vocab(self.data, self.index2token)

    def to_dgl_graph(self):
        """ Converts BagsWithVocab to dgl (dgl.ai) graph """
        import dgl
        G = dgl.DGLGraph()
        print("Constructing bipartite graph between", len(self.data), "users.")
        n_users = len(self.data)
        n_items = len(self.vocab)
        # TODO Add user data (paper title etc)
        G.add_nodes(n_users)
        G.add_nodes(n_items)
        for i, dsts in tqdm(enumerate(self.data)):
            for j in dsts:
                # n_users offset because graph is bipartite!
                G.add_edge(i, n_users + j)
        
        return G





if __name__ == '__main__':
    import doctest
    doctest.testmod()
