import torch
import torch.nn as nn
from docutils.nodes import inline

from torch import optim

from abc import ABC, abstractmethod
from collections import OrderedDict, Counter
import itertools as it
import torch
import scipy.sparse as sp
import numpy as np



from .ub import GensimEmbeddedVectorizer
"""
Key idea: The conditions we pass through all the code
could be a list of (name, condition_obj) tuples.
Each condition_obj has an interface to encode a batch
and (optional) to update its parameters wrt (global) ae loss.




"""


def _check_conditions(conditions, condition_data):
    """ Checks condition list and condition data for validity.
    Arguments
    =========
    conditions: a condition list instance
    condition_data: condition data that should correspond to conditions

    Returns
    =======
    use_condition:
        - True if conditions are present and condition_data matches,
        - False if neither conditions nor condition_data is supplied.

    Raises
    ======
    AssertionError, when `conditions` does not match with `condition_data`

    """
    if not conditions and not condition_data:
        # Neither supplied, do not use conditions
        return False

    assert isinstance(conditions, ConditionList), "`conditions` no instance of ConditionList"
    assert condition_data and conditions, "Mismatch between condition spec and supplied condition data."
    assert len(condition_data) == len(conditions), "Unexpected number of supplied condition data"

    return True

class ConditionList(OrderedDict):
    """
    Condition list is an ordered dict with attribute names as keys and
    condition instances as values.
    Order is meaningful.
    It subclasses OrderedDict.
    """

    def __init__(self, items):
        super(ConditionList, self).__init__(items)
        assert all(isinstance(v, ConditionBase) for v in self.values())

    def fit(self, raw_inputs):
        """ Fits all conditions to data """
        assert len(raw_inputs) == len(self)
        for cond, cond_inp in zip(self.values(), raw_inputs):
            cond.fit(cond_inp)
        return self

    def transform(self, raw_inputs):
        """ Transforms `raw_inputs` with all conditions """
        assert len(raw_inputs) == len(self)
        return [c.transform(inp) for c, inp in zip(self.values(), raw_inputs)]

    def fit_transform(self, raw_inputs):
        """ Forwards to fit_transform of all conditions,
        returns list of transformed condition inputs"""
        assert len(raw_inputs) == len(self)
        return [cond.fit_transform(inp) for cond, inp
                in zip(self.values(), raw_inputs)]

    def encode_impose(self, x, condition_inputs, dim=None):
        """ Subsequently conduct encode & impose with all conditions
        in order.
        : param x: ???, suspect its the normal data not the condition ones
        """
        assert len(condition_inputs) == len(self)
        for condition, condition_input in zip(self.values(), condition_inputs):
            x = condition.encode_impose(x, condition_input, dim)
        return x

    def encode(self, condition_inputs):
        assert len(condition_inputs) == len(self)
        return [condition.encode(condition_input) for condition, condition_input
                in zip(self.values(), condition_inputs)]


    def zero_grad(self):
        """ Forward the zero_grad call to all conditions in list
        such they can reset their gradients """
        for condition in self.values():
            condition.zero_grad()
        return self

    def step(self):
        """ Forward the step call to all conditions in list,
        such that these can update their individual parameters"""
        for condition in self.values():
            condition.step()
        return self

    def size_increment(self):
        """ Aggregates sizes from various conditions
        for convenience use in determining decoder properties
        """
        return sum(v.size_increment() for v in self.values())

    def train(self):
        # Put all modules into train mode, if they has such a method
        for condition in self.values():
            if hasattr(condition, 'train'):
                condition.train()

    def eval(self):
        # Put all modules into train mode, if they have such a method
        for condition in self.values():
            if hasattr(condition, 'eval'):
                condition.eval()


class ConditionBase(ABC):
    """ Abstract Base Class for a generic condition """

    #####################################################################
    # Condition supplies info how much it will increment the size of data
    # Some conditions might want to prepare on whole dataset .
    # eg to build a vocabulary and compute global IDF and stuff.
    # Thus, conditions may implement fit and transform methods.
    # Fit adapts the condition object to the data it will receive.
    # Transform may apply some preprocessing that can be conducted globally
    # once.
    def fit(self, raw_inputs):
        """ Prepares the condition wrt to the whole raw data for the condition
        To be called *once* on the whole (condition)-data.
        """
        return self

    def transform(self, raw_inputs):
        """ Returns transformed raw_inputs, can be applied globally as
        preprocessing step """
        return raw_inputs

    def fit_transform(self, raw_inputs):
        """ Fit to `raw_inputs`, then transform `raw_inputs`. """
        return self.fit(raw_inputs).transform(raw_inputs)

    # Latest after preparing via fit,
    # size_increment should yield reasonable results.
    @abstractmethod
    def size_increment(self):
        """ Returns the output dimension of the condition,
        such that:
        code.size(1) + condition.size_increment() = conditioned_code.size(1)
        Note that for additive or multiplicative conditions,
        size_increment should be zero.
        """
    #####################################################################

    ###########################################################################
    # Condition can encode the raw input and knows how to impose itself to data
    def encode(self, inputs):
        """ Encodes the input for the condition """
        return inputs

    @abstractmethod
    def impose(self, inputs, encoded_condition, dim=None):
        """ Applies the condition, for instance by concatenation.
        Could also use multiplicative or additive conditioning.
        """
        raise NotImplementedError

    def encode_impose(self, inputs, condition_input, dim=None):
        """ First encodes `condition_input`, then applies condition to `inputs`.
        """
        return self.impose(inputs, self.encode(condition_input), dim=None)
    ###########################################################################

    ################################################
    # Condition knows how to optimize own parameters
    def zero_grad(self):
        """
        Clear out gradients.
        Per default does nothing on step
        (optional for subclasses to implement).
        To be called before each batch.
        """
        return self

    def step(self):
        """
        Update condition's associated parameters.
        Per default does nothing on step (optional for subclasses to implement.

        To be called after each batch.
        """
        return self
    ################################################

    ################################################
    # Condition knows how to be in train / eval mode
    def train(self):
        """
        Put into training mode.
        Per default does nothing.
        To be called before training.
        """
        return self

    def eval(self):
        """
        Put into evaluation mode
        Per default does nothing.
        To be called before evaluation.
        """
        return self
    ################################################


    @classmethod
    def __subclasshook__(cls, C):
        if cls is ConditionBase:
            # Check if abstract parts of interface are satisified
            mro = C.__mro__
            if all([any("encode" in B.__dict__ for B in mro),
                    any("impose" in B.__dict__ for B in mro),
                    any("encode_impose" in B.__dict__ for B in mro),
                    any("size_increment" in B.__dict__ for B in mro),
                    any("fit" in B.__dict__ for B in mro),
                    any("transform" in B.__dict__ for B in mro),
                    any("fit_transform" in B.__dict__ for B in mro),
                    any("zero_grad" in B.__dict__ for B in mro),
                    any("step" in B.__dict__ for B in mro),
                    any("train" in B.__dict__ for B in mro),
                    any("eval" in B.__dict__ for B in mro)]):
                return True
        return NotImplemented  # Proceed with usual mechanisms


""" Three basic variants of conditioning
1. Concatenation-based conditioning
2. Conditional biasing
3. Conditional scaling
See also: https://distill.pub/2018/feature-wise-transformations/

Condition implementations should subclass one of the following three baseclasses.
"""


class ConcatenationBasedConditioning(ConditionBase):
    """
    A `ConditionBase` subclass to implement concatenation based conditioning.
    """
    # Subclasses still need to specify .size_increment()
    # as concatenation based
    dim = 1

    @abstractmethod
    def size_increment(self):
        """ Subclasses need to specify size increment """

    def impose(self, inputs, encoded_condition, dim=None):
        """ Concat condition at specified dimension (default 1) """
        if dim is None:
            dim = self.dim
        return torch.cat([inputs, encoded_condition], dim=dim)


class ConditionalBiasing(ConditionBase):
    """
    A `ConditionBase` subclass to implement conditional biasing
    """
    def impose(self, inputs, encoded_condition, dim=None):
        """ Applies condition by addition """
        return inputs + encoded_condition

    def size_increment(self):
        """ Biasing does not increase vector size """
        return 0


class ConditionalScaling(ConditionBase):
    """
    A `ConditionBase` subclass to implement conditional scaling
    """
    def impose(self, inputs, encoded_condition, dim=None):
        """ Applies condition by multiplication """
        return inputs * encoded_condition

    def size_increment(self):
        """ Scaling does not increase vector size """
        return 0


class PretrainedWordEmbeddingCondition(ConcatenationBasedConditioning):
    """ A concatenation-based condition using a pre-trained word embedding """

    def __init__(self, vectors, dim=1, use_cuda=torch.cuda.is_available(), **tfidf_params):
        self.vect = GensimEmbeddedVectorizer(vectors, **tfidf_params)
        self.dim = dim
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

    def fit(self, raw_inputs):
        self.vect.fit(raw_inputs)
        return self

    def transform(self, raw_inputs):
        return self.vect.transform(raw_inputs)

    def fit_transform(self, raw_inputs):
        return self.vect.fit_transform(raw_inputs)

    def encode(self, inputs):
        # GensimEmbeddedVectorizer yields numpy array
        return torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

    def size_increment(self):
        # Return embedding dimension
        return self.vect.embedding.shape[1]


class EmbeddingBagCondition(ConcatenationBasedConditioning):
    """ A condition with a *trainable* embedding bag.
    """
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        self.embedding_bag = nn.EmbeddingBag(num_embeddings,
                                             embedding_dim,
                                             **kwargs)
        self.optimizer = torch.optim.Adam(self.embedding_bag.parameters())
        self.embedding_dim = embedding_dim

    def encode(self, inputs):
        return self.embedding_bag(inputs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # loss.backward() to be called before by client (such as in ae_step)
        # The condition object can update its own parameters wrt global loss
        self.optimizer.step()

    def size_increment(self):
        return self.embedding_dim


class CategoricalCondition(ConcatenationBasedConditioning):
    """ A *trainable* condition for categorical attributes.
    """
    padding_idx = 0

    def __init__(self, embedding_dim, vocab_size=None,
                 sparse=True,
                 use_cuda=torch.cuda.is_available(),
                 embedding_on_gpu=False,
                 lr=1e-3,
                 reduce=None,
                 **embedding_params):
        """
        Arguments
        ---------
        - embedding_dim: int - Size of the embedding
        - vocab_size: int - Vocabulary size limit (if given)
        - ignore_oov: bool - If given, set oov embedding to zero
        - lr: float - initial learning rate for Adam / SparseAdam
        - sparse: bool - If given, use sparse embedding & optimizer
        - reduce: None or str - if given, expect list-of-list like inputs
                  and aggregate according to `reduce` in 'mean', 'sum', 'max'
        """
        # register this module's parameters with the optimizer
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab = None
        self.embedding = None
        self.optimizer = None
        self.lr = lr

        # Optimization and memory storay
        self.sparse = sparse
        self.use_cuda = use_cuda
        self.embedding_on_gpu = embedding_on_gpu

        # We take care of vocab handling & padding ourselves
        assert "padding_idx" not in embedding_params, "Padding is fixed with token 0"
        self.embedding_params = embedding_params


        assert reduce is None or reduce in ['mean','sum','max'], "Reduce neither None nor in 'mean','sum','max'"
        self.reduce = reduce

    def fit(self, raw_inputs):
        """ Learn a vocabulary """
        flat_items = raw_inputs if self.reduce is None else list(it.chain.from_iterable(raw_inputs))
        if self.vocab_size is None:
            # if vocab size is None, use all items
            cutoff = len(flat_items)
        if isinstance(self.vocab_size, float):
            # if vocab size is float, interprete it as percentage of top items (authors)
            cutoff = int(self.vocab_size * len(flat_items))
        else:
            # else use fixed vocab size or None, which is fine aswell
            cutoff = int(self.vocab_size)
        print("Using top {:.2f}% authors ({})".format(cutoff / len(flat_items) * 100, cutoff))

        item_cnt = Counter(flat_items).most_common(cutoff)
        # index 0 is reserved for unk idx
        self.vocab = {value: idx + 1 for idx, (value, __) in enumerate(item_cnt)}
        num_embeddings = len(self.vocab) + 1
        self.embedding = nn.Embedding(num_embeddings,
                                      self.embedding_dim,
                                      padding_idx=self.padding_idx,
                                      **self.embedding_params,
                                      sparse=self.sparse)
        if self.use_cuda and self.embedding_on_gpu:
            # Put the embedding on GPU only when wanted
            self.embedding = self.embedding.cuda()
        if self.sparse:
            self.optimizer= optim.SparseAdam(self.embedding.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.embedding.parameters(), lr=self.lr)
        return self

    def transform(self, raw_inputs):
        # Actually np.array is not needed,
        # else we would need to do the padding globally
        if self.reduce is None:
            return [self.vocab.get(x, self.padding_idx) for x in raw_inputs]
        else:
            return [[self.vocab.get(x, self.padding_idx) for x in l] for l in raw_inputs]

    def _pad_batch(self, batch_inputs):
        maxlen = max(len(l) for l in batch_inputs)
        return [l + [self.padding_idx] * (maxlen - len(l)) for l in batch_inputs]

    def encode(self, inputs):
        if self.reduce is not None:
            # inputs may have variable lengths, pad them
            inputs = self._pad_batch(inputs)
        inputs = torch.tensor(inputs, device=self.embedding.weight.device)
        h = self.embedding(inputs)
        if self.reduce is not None:
            # self.reduce in ['mean','sum','max']
            h = getattr(h, self.reduce)(1)
        if self.use_cuda:
            h = h.cuda()
        return h

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # loss.backward() to be called before by client (such as in ae_step)
        # The condition object can update its own parameters wrt global loss
        self.optimizer.step()

    def size_increment(self):
        return self.embedding_dim



# idk whether the following is helpful in the end.

class Condition(ConditionBase):
    """ A generic condition class.
    Arguments
    ---------
    - encoder: callable, nn.Module
    - preprocessor: object satisfying fit, transform, fit_transform
    - optimizer: optimizer satisfying step, zero_grad, makes sense to operate
      on encoder's parameters
    - size_increment: int - When in concat mode, how much does this condition
      attach
    - dim: int - When in concat mode, to which dim should this condition
      concatenate


    """
    def __init__(self, preprocessor=None, encoder=None, optimizer=None,
                 mode="concat", size_increment=0, dim=1):
        if encoder is not None:
            assert callable(encoder)
        assert mode in ["concat", "bias", "scale"]
        if mode == "concat":
            assert size_increment > 0, "Specify size increment in concat mode"
        else:
            assert size_increment == 0,\
                "Size increment should be zero in bias or scale modes"
        if preprocessor is not None:
            assert hasattr(preprocessor, 'fit'),\
                "Preprocessor has no fit method"
            assert hasattr(preprocessor, 'transform'),\
                "Preprocessor has no transform method"
            assert hasattr(preprocessor, 'fit_transform'),\
                "Preprocessor has no fit_transform method"
        if optimizer is not None:
            assert hasattr(optimizer, 'zero_grad')
            assert hasattr(optimizer, 'step')
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.optimizer = optimizer
        self.mode_ = mode
        self.dim = dim

    def fit(self, raw_inputs):
        if self.preprocessor is not None:
            self.preprocessor.fit(raw_inputs)
        return self

    def transform(self, raw_inputs):
        if self.preprocessor is not None:
            x = self.preprocessor.transform(raw_inputs)
        return x

    def fit_transform(self, raw_inputs):
        if self.preprocessor is not None:
            x = self.preprocessor.fit_transform(raw_inputs)
        return x

    def encode(self, inputs):
        if self.encoder is not None:
            return self.encoder(inputs)
        return inputs

    def impose(self, inputs, encoded_condition, dim=None):
        if self.mode_ == "concat":
            out = torch.cat([inputs, encoded_condition], dim=self.dim)
        elif self.mode_ == "bias":
            out = inputs + encoded_condition
        elif self.mode_ == "scale":
            out = inputs * encoded_condition
        else:
            raise ValueError("Unknown mode: " + self.mode_)
        return out

    def size_increment(self):
        return self.size_increment

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def train(self):
        if self.encoder is not None:
            self.encoder.train()

    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
