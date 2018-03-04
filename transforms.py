#!/usr/bin/env python
""" This module contains transforming functions """

from functools import reduce
import scipy.sparse as sp
import numpy as np


def star(fn):
    """
    Allows other functions to deal with multiple arguments.
    >>> f =lambda x: x + 1
    >>> f(0)
    1
    >>> star(f)(1,2)
    (2, 3)
    """
    return lambda *input: tuple(map(fn, input))


def pipe(*transforms):
    """
    Left-to-right execution of transforming functions
    >>> pipe = pipe(lambda x: x+1, lambda x: x*2)
    >>> pipe(0)
    2
    """
    return compose(*reversed(transforms))


def compose(*functions):
    """
    Composition of functions
    >>> f = lambda x: x + 1
    >>> g = lambda x: x * 2
    >>> c = compose(f,g)
    >>> c(0)
    1
    >>> compose(g,f)(0)
    2
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def sparse2lists(input):
    """
    Transforms sparse matrix into list of lists.
    Only the indices of nonzero entries are considered, not their values.
    >>> import numpy, scipy.sparse
    >>> A = numpy.diag(numpy.ones([3]))
    >>> A[2,0] = 7.0
    >>> A
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 7.,  0.,  1.]])
    >>> C = scipy.sparse.csr_matrix(A)
    >>> sparse2lists(C)
    [[0], [1], [0, 2]]
    """
    assert hasattr(input, 'shape') and hasattr(input, 'nonzero')
    assert len(input.shape) == 2
    lists = [[] for _ in range(input.shape[0])]
    for i, j in zip(*input.nonzero()):
        lists[int(i)].append(int(j))

    return lists


def lists2indices(input):
    """
    Extracts typical rows, cols from list of lists
    >>> lists = [[1,2], [2,3], [0,1]]
    >>> lists2indices(lists)
    ((0, 0, 1, 1, 2, 2), (1, 2, 2, 3, 0, 1))
    >>> lists = [[], [1], [1,2], []]
    >>> lists2indices(lists)
    ((1, 2, 2), (1, 1, 2))
    >>> lists = [[], []]
    >>> lists2indices(lists)
    ((), ())

    """
    ind = [(r, c) for r, cs in enumerate(input) for c in cs]
    if not ind:
        return ((), ())
    else:
        return tuple(zip(*ind))


def _check_shape(input, size):
    """ Convenient usage of shape attributes.
    It is also ok to just provide 2nd dimension
    >>> bags = [[1],[2],[3], [27]]
    >>> _check_shape(bags, 3)
    (4, 3)
    >>> _check_shape(bags, [4, 3])
    (4, 3)
    >>> _check_shape(bags, [-1, 3])
    (4, 3)
    >>> _check_shape(bags, [None, 3])
    (4, 3)
    """

    if isinstance(size, int):
        # dim
        return len(input), size
    # might be too relaxed
    # elif len(size) == 1:
    #     return len(input), int(size[0])
    elif len(size) == 2:
        if size[0] is None or size[0] == -1:
            # [-1, dim] or [None, dim]
            return len(input), size[1]
        else:
            # [N, dim]
            assert len(input) == size[0]
            return tuple(size)
    else:
        raise ValueError("Incorrect Shape")


def lists2sparse(input, size):
    """
    Transforms a list of lists into sparse coo format.
    Shape may not be ommitted.
    >>> bags = [[0], [1], [0, 2]]
    >>> X = lists2sparse(bags, (3,3))
    >>> X.toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  1.]])
    """
    shape = _check_shape(input, size)
    ind = lists2indices(input)
    v = np.ones(len(ind[0]))
    sparse = sp.coo_matrix((v, ind), shape=shape)
    return sparse


def lists2dense(input, size, zero_gen=np.zeros):
    """
    Arguments
    =========

    input : set of sets
    size : desired dimensionality
    zero_gen: a function (shape) -> zero array
    >>> bags = [[0], [1], [0, 2]]
    >>> (lists2sparse(bags, (3,3)).toarray() == lists2dense(bags, (3,3))).all()
    True
    >>> import torch
    >>> lists2dense(bags, 4, torch.zeros)
    <BLANKLINE>
     1  0  0  0
     0  1  0  0
     1  0  1  0
    [torch.FloatTensor of size 3x4]
    <BLANKLINE>
    """
    shape = _check_shape(input, size)
    A = zero_gen(shape)
    ind = lists2indices(input)
    # TODO FIXME IndexError when executing recursive.py
    A[ind] = 1.0
    return A


def to_strings(input):
    """
    >>> x = [[3, 4, 2], [1, 2], [], [1, 1, 1, 1]]
    >>> to_strings(x)
    [['3', '4', '2'], ['1', '2'], [], ['1', '1', '1', '1']]
    """
    return [[str(tok) for tok in bag] for bag in input]


def padded_sequence(lists,
                    pad_with=0,
                    sort=True,
                    batch_first=False):
    import torch
    if sort:
        lists = sorted(lists, key=len, reverse=True)
    seq_lengths = torch.LongTensor(list(map(len, lists)))
    seq_tensor = torch.LongTensor(len(lists), seq_lengths.max())\
        .fill_(pad_with)
    for idx, (seq, seqlen) in enumerate(zip(lists, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    if not batch_first:
        seq_tensor = seq_tensor.transpose(0, 1)

    return seq_tensor, seq_lengths


def ToSparseTensor(input, dim=None):
    import torch
    if isinstance(input, torch.sparse.FloatTensor):
        return input
    elif sp.issparse(input):  # any of scipy.sparse
        input = input.tocoo()
        i = np.vstack([input.row, input.col])
        print(i)
        i = torch.from_numpy(i.astype(np.int64))
        v = torch.FloatTensor(input.data)
        output = torch.sparse.FloatTensor(i, v, torch.Size(input.shape))
    elif isinstance(input, np.ndarray):  # nd array
        assert len(input.shape) == 2, "Invalid input shape: " + input.shape
        nz = np.nonzero(input)
        i = torch.LongTensor(np.vstack(*nz))
        v = torch.FloatTensor(input[nz])
        output = torch.sparse.FloatTensor(i, v, torch.Size(input.shape))
    else:  # assume list of list
        assert dim is not None, "Dimensions undefined"
        ind = []
        for row, tokens in enumerate(input):
            ind.extend([(row, int(token)) for token in tokens])
        i = torch.LongTensor(ind).t()
        v = torch.ones(len(ind))
        output = torch.sparse.FloatTensor(i, v, torch.Size([len(input), dim]))
    return output


def ToTensor(input):
    import torch
    if isinstance(input, torch.Tensor):
        return input
    input = input.toarray() if sp.issparse(input) else input
    tensor = torch.from_numpy(input.astype(np.float32))
    return tensor


if __name__ == '__main__':
    import doctest
    doctest.testmod()
