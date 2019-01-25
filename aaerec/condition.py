import torch
import torch.nn as nn

from torch import optim

from abc import ABC, abstractmethod
from collections import OrderedDict
import torch

from .ub import GensimEmbeddedVectorizer
"""
Key idea: The conditions we pass through all the code
could be a list of (name, condition_obj) tuples.
Each condition_obj has an interface to encode a batch
and (optional) to update its parameters wrt (global) ae loss.




"""


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
        assert len(raw_inputs) == len(self)
        return [cond.fit_transform(inp) for cond, inp
                in zip(self.values(), raw_inputs)]

    def encode_impose(self, x, condition_inputs):
        """ Subsequently conduct encode & impose with all conditions
        in order.
        """
        assert len(condition_inputs) == len(self)
        for condition, condition_input in zip(self.values(), condition_inputs):
            x = condition.encode_impose(x, condition_input)
        return x

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

    # Latest after preparing, size_increment should yield reasonable results.
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
    @abstractmethod
    def encode(self, inputs):
        """ Encodes the input for the condition """

    @abstractmethod
    def impose(self, inputs, encoded_condition):
        """ Applies the condition, for instance by concatenation.
        Could also use multiplicative or additive conditioning.
        """

    def encode_impose(self, inputs, condition_input):
        """ First encodes `condition_input`, then applies condition to `input`.
        """
        return self.impose(input, self.encode(condition_input))
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
                    any("step" in B.__dict__ for B in mro)]):
                return True
        return NotImplemented  # Proceed with usual mechanisms


""" Three basic variants of conditioning
1. Concatenation-based conditioning
2. Conditional biasing
3. Conditional scaling
See also: https://distill.pub/2018/feature-wise-transformations/

Condition implementations should subclass one of the following three baseclasses.
"""

class ConcatenationBasedConditioningMixin():
    # Subclasses still need to specify .size_increment()
    # as concatenation based
    dim = 1

    def impose(self, inputs, encoded_condition):
        """ Concat condition at specified dimension (default 1) """
        return torch.cat([input, encoded_condition], dim=self.dim)


class ConditionalBiasingMixin():
    """
    A Mixin for ConditionBase's subclasses to implement conditional biasing
    """
    def impose(self, inputs, encoded_condition):
        """ Applies condition by addition """
        return inputs + encoded_condition

    @property
    def size_increment(self):
        """ Biasing does not increase vector size """
        return 0


class ConditionalScalingMixin():
    """
    A Mixin for ConditionBase's subclasses to implement conditional scaling
    """
    def impose(self, input, encoded_condition):
        """ Applies condition by multiplication """
        return input * encoded_condition

    def size_increment(self):
        """ Scaling does not increase vector size """
        return 0


class PretrainedWordEmbeddingCondition(
        GensimEmbeddedVectorizer, # fit & transform
        ConditionBase,
        ConcatenationBasedConditioningMixin):
    def __init__(self, vectors, **tfidf_params):
        GensimEmbeddedVectorizer.__init__(vectors, **tfidf_params)

    def encode(self, wv_centroids):
        """ Transformation is conducted globally,
        so we only pass through here.
        """
        return wv_centroids

    def size_increment(self):
        # Return embedding dimension
        return self.embedding.shape[1]


class EmbeddingBagCondition(ConditionBase, ConcatenationBasedConditioningMixin):
    """ A condition with a *trainable* embedding bag.
    It is suited for conditioning on categorical variables.
    >>> cc = EmbeddingBagCondition(100,10)
    >>> cc
    EmbeddingBagCondition(
      (embedding_bag): EmbeddingBag(100, 10, mode=mean)
    )
    >>> issubclass(EmbeddingBagCondition, ConditionBase)
    True
    >>> isinstance(cc, ConditionBase)
    True
    """

    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        """TODO: to be defined1. """
        self.embedding_bag = nn.EmbeddingBag(num_embeddings,
                                             embedding_dim,
                                             **kwargs)

        # register this module's parameters with the optimizer
        self.optimizer = optim.Adam(self.embedding_bag.parameters())
        self.output_dim = embedding_dim

    def encode(self, input):
        return self.embedding_bag(input)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # loss.backward() to be called before by client (such as in ae_step)
        # The condition object can update its own parameters wrt global loss
        self.optimizer.step()

    def size_increment(self):
        return self.output_dim
