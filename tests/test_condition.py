""" Tests various functionalities wrt conditions """
import pytest
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from aaerec.condition import EmbeddingBagCondition,\
    PretrainedWordEmbeddingCondition,\
    ConditionBase,\
    ConcatenationBasedConditioning,\
    ConditionalBiasing,\
    ConditionalScaling,\
    CategoricalCondition,\
    Condition,\
    ConditionList


def test_condition_abc():

    assert issubclass(ConcatenationBasedConditioning, ConditionBase)
    assert issubclass(ConditionalBiasing, ConditionBase)
    assert issubclass(ConditionalScaling, ConditionBase)

    assert issubclass(EmbeddingBagCondition, ConditionBase)
    assert issubclass(PretrainedWordEmbeddingCondition, ConditionBase)


def test_condition_simple():
    code = torch.rand(100, 10)
    # 2 random values with vocabulary 0, 1 per sample
    c_batch = (torch.rand(100, 2) < 0.5).long()

    ebc = EmbeddingBagCondition(2, 10)

    assert isinstance(ebc, ConditionBase)
    assert isinstance(ebc, ConcatenationBasedConditioning)

    condition_encoded = ebc.encode(c_batch)

    conditioned_code = ebc.impose(code, condition_encoded)

    # Dim 1 is expected to increase by condition size
    assert conditioned_code.size(1) == code.size(1) + ebc.size_increment()

    # Batch dim should not be changed
    assert code.size(0) == conditioned_code.size(0)


def test_condition_list():
    """ Test list of conditions """
    code = torch.rand(100, 10)
    c1_batch = (torch.rand(100, 2) < 0.5).long()
    c2_batch = (torch.rand(100, 2) < 0.5).long()

    ebc1 = EmbeddingBagCondition(2, 10)
    ebc2 = EmbeddingBagCondition(2, 10)

    condition_list = ConditionList([
        ('title', ebc1),
        ('something', ebc2)
    ])

    # keys of attributes should be accessible from cond list
    assert list(condition_list.keys()) == ['title', 'something']
    # the actual conditions are also accessible as a whole..
    assert list(condition_list.values()) == [ebc1, ebc2]
    # .. or individually by name
    assert condition_list['title'] == ebc1
    assert condition_list['something'] == ebc2

    assert condition_list.size_increment() == 20

    conditioned_code = condition_list.encode_impose(code, [c1_batch, c2_batch])

    assert conditioned_code.size(1) \
        == code.size(1) + condition_list.size_increment()

    assert code.size(0) == conditioned_code.size(0)


def test_optim_step_callback():
    """ Test zero_grad / step optimization """
    code = torch.rand(100, 10)
    # 2 random values with vocabulary 0, 1 per sample
    c_batch = (torch.rand(100, 2) < 0.5).long()
    ebc = EmbeddingBagCondition(2, 10)

    target = torch.zeros(20)

    losses = []
    for _ in range(2):
        # dummy training loop
        ebc.zero_grad()
        criterion = torch.nn.MSELoss()
        conditioned_code = ebc.encode_impose(code, c_batch)
        loss = criterion(conditioned_code, target)
        loss.backward()
        losses.append(loss.item())
        ebc.step()

    assert len(losses) == 2
    # One loss improvement should be possible
    assert losses[1] < losses[0]


def test_word_emb_condition():
    import gensim
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "the cat sits on the mat",
        "if it fits, I sits"
    ]
    emb_dim = 10
    model = gensim.models.word2vec.Word2Vec(
        [s.split() for s in sentences],
        min_count=1, window=2, size=emb_dim
    )
    condition = PretrainedWordEmbeddingCondition(model.wv, use_cuda=False)
    sentences_trf = condition.fit_transform(sentences)

    code = torch.rand(len(sentences), 5)
    conditioned_code = condition.encode_impose(code, sentences_trf)

    assert conditioned_code.size(1) == code.size(1) + condition.size_increment()


def test_full_pipeline():
    """ This test shows how to use condition (-list) in a complete pipeline """
    import gensim
    data = {
        'titles':  [
            "the quick brown fox jumps over the lazy dog",
            "the cat sits on the mat",
            "if it fits, I sits"
        ],
        'authors': [
            "Iacopo",
            "Gunnar",
            "Lukas",
        ]
    }

    emb_dim = 10
    model = gensim.models.word2vec.Word2Vec(
        [s.split() for s in data['titles']],
        min_count=1, window=2, size=emb_dim
    )
    cond1 = PretrainedWordEmbeddingCondition(model.wv, use_cuda=False)
    cond2 = CategoricalCondition(emb_dim, vocab_size=3, use_cuda=False)

    clist = ConditionList([('titles', cond1),
                           ('authors', cond2)])

    # Apply fit_transform on all conditions, store results
    prepped_inputs = clist.fit_transform([data[k] for k in clist.keys()])

    # Let's assume the encoder produced these codes originally
    codes = torch.rand(3, 10)

    criterion = torch.nn.MSELoss()

    decoder = torch.nn.Linear(codes.size(1) + clist.size_increment(),
                              codes.size(1))
    optimizer = torch.optim.Adam(decoder.parameters())

    losses = []
    for __epoch in range(10):
        for start in range(len(codes)):
            end = start + 1
            code_batch = codes[start:end]
            # Batch all condition inputs
            cinputs_batch = [inp[start:end] for inp in prepped_inputs]
            clist.zero_grad()
            optimizer.zero_grad()
            conditioned_code = clist.encode_impose(code_batch, cinputs_batch)
            # assert dim is predictable for decoder
            assert conditioned_code.size(1) - code_batch.size(1)\
                == clist.size_increment()
            out = decoder(conditioned_code)
            # Reconstruction loss
            loss = criterion(out, code_batch)
            loss.backward()
            losses.append(loss.item())
            clist.step()
            optimizer.step()


def test_categorical_condition():

    authors = ["Mr X", "Falafel", "Pizza", "Am I hungry?", "Mr X"]
    catcond = CategoricalCondition(20, vocab_size=10, use_cuda=False)
    author_ids = catcond.fit_transform(authors)
    some_code = torch.rand(len(authors), 10)

    encoded_authors = catcond.encode(author_ids)
    assert encoded_authors.size(0) == len(authors) and encoded_authors.size(1) == 20

    # Mr X is Mr X
    assert ((encoded_authors[0] - encoded_authors[-1]).abs() < 1e-8).all()

    cond_code = catcond.impose(some_code, encoded_authors)
    # 20 + 10 should turn out to be 30
    assert cond_code.size(0) == len(authors) and cond_code.size(1) == 30


def test_categorical_condition_unk_treatment():
    authors = ["A", "A", "B", "B", "C"]


    ## OOV SHOULD BE IGNORE
    catcond = CategoricalCondition(20, vocab_size=2, use_cuda=False)
    # Vocab should only hold A and B
    author_ids = catcond.fit_transform(authors)
    assert author_ids[-1] == 0

    enc_authors = catcond.encode(author_ids)
    # C token should be zero
    assert ((enc_authors[-1] - torch.zeros(20)).abs() < 1e-8).all()

    with pytest.raises(AssertionError):
        catcond = CategoricalCondition(12, use_cuda=False, padding_idx=1231)



def test_categorical_condition_sparse():
    authors = ["A", "A", "B", "B", "C"]
    catcond = CategoricalCondition(20, use_cuda=False,
                                   sparse=True)

    author_ids = catcond.fit_transform(authors)

    enc_authors_1 = catcond.encode(author_ids)

    loss = torch.nn.functional.mse_loss(enc_authors_1, torch.zeros(5,20))

    catcond.zero_grad()
    loss.backward()
    catcond.step()

    # Encoded authors should now be closer to zero
    enc_authors_2 = catcond.encode(author_ids)

    assert (enc_authors_2.abs().sum() < enc_authors_1.abs().sum()).all()

def test_categorical_condition_listoflists():
    authors = [["A","B"],
               ["A", "C"],
               ["B", "C"],
               ["A"],
               ["B"],
               ["A", "B", "C"]]
    catcond = CategoricalCondition(20, use_cuda=False,
                                   sparse=True, reduce='mean')

    author_ids = catcond.fit_transform(authors)

    enc_authors_1 = catcond.encode(author_ids)

    loss = torch.nn.functional.mse_loss(enc_authors_1, torch.zeros(len(authors),20))

    catcond.zero_grad()
    loss.backward()
    catcond.step()

    # Encoded authors should now be closer to zero
    enc_authors_2 = catcond.encode(author_ids)

    assert (enc_authors_2.abs().sum() < enc_authors_1.abs().sum()).all()

def test_categorical_condition_with_sklearn_shuffle():
    authors = [["A","B"],
               ["A", "C"],
               ["B", "C"],
               ["A"],  # 3
               ["B"],  # 4
               ["A", "B", "C"]]
    catcond = CategoricalCondition(20, use_cuda=False,
                                   sparse=True, reduce='mean')

    author_ids = catcond.fit_transform(authors)

    labels = np.arange(len(authors))
    
    labels_shf, authors_shf, author_ids_shf = shuffle(labels, authors, author_ids)

    for l, a, i in zip(labels_shf, authors_shf, author_ids_shf):
        if l == 0:
            assert a[0] == "A" and a[1] == "B"
            assert i[0] == catcond.vocab[a[0]]
            assert i[1] == catcond.vocab[a[1]]
        if l == 3:
            assert a[0] == "A"
            assert i[0] == catcond.vocab[a[0]]
        if l == 4:
            assert a[0] == "B"
            assert i[0] == catcond.vocab[a[0]]



def test_assemble_condition():
    documents = [
        "Spam Spam Spam",
        "Ham Ham Ham and cheese",
        "Cookies cookies",
        "Nothing else but cookies",
        "Cheese",
        "am I hungry",
        "Cookies Spam",
        "Cheese Spam",
        "I Spam"
    ]
    labels = torch.tensor([1, 0, 0, 0, 0, 0, 1, 1, 1])

    tfidf = TfidfVectorizer()
    # 10 distinct words, 2 classes
    encoder = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(encoder.parameters())
    def enc_fn(x):
        return encoder(torch.FloatTensor(x.toarray()))
    condition = Condition(tfidf, enc_fn, optimizer, size_increment=2)

    criterion = torch.nn.CrossEntropyLoss()

    for __ in range(3):
        x_trf = condition.fit_transform(documents)
        x_enc = condition.encode(x_trf)
        loss = criterion(x_enc, labels)
        loss.backward()
        condition.step()

