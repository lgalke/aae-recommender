import torch
import torch.nn as nn

from torch import optim

"""
Key idea: The conditions we pass through all the code
could be a list of (name, condition_obj) tuples.
Each condition_obj has an interface to encode a batch
and (optional) to update its parameters wrt (global) ae loss.


""" 

# TODO FIXME this is an implementation *draft*

class ConditionBase():


    # TODO: find a way to make class itself OR list of it easyly iterable --> backwards compatible
    # TODO: ~ returning attribute name upon call
    # TODO: substitute workaround of condtion.condition_name = "playlist name" / "description tokens"
    condition_name = None
    def condition_name(self):
        return self.condition_name

    def __init__(self,condition_name):
        raise NotImplementedError("NotImplemented: trying to call abstract method")

    def encode(self, input):
        """ Encodes the input """
        raise NotImplementedError("NotImplemented: trying to call abstract method")

    def step(self):
        """ (optional) Updates its own parameters """
        pass


class PretrainedWordEmbeddingCondition(ConditionBase):
    def __init__(self, gensim_embedding, **tfidf_params):
        self.vect = EmbeddedVectorizer(embedding, **self.tfidf_params)

    def encode(self, input):
        return self.vect.fit_transform(input)

    def step(self):
        pass


class CategoricalCondition(nn.Module, ConditionBase):

    """Docstring for Condition. """

    def __init__(self, n_inputs, n_outputs):
        """TODO: to be defined1. """
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(n_inputs, n_outputs)

        # register this module's parameters with the optimizer
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, input):
        # in this case embedding, but can be anything
        return self.embedding(input)

    def encode(self, input):
        return self.forward(input)

    def step(self):
        # loss.backward() should be called before by client (such as in ae_step)
        # Here the condition object can update its own parameters wrt global loss
        self.optimizer.step()





        
