""" Tests various functionalities wrt conditions """
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
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
    condition = PretrainedWordEmbeddingCondition(model.wv)
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
    cond1 = PretrainedWordEmbeddingCondition(model.wv)
    cond2 = CategoricalCondition(3, emb_dim)

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




