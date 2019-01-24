import torch
from aaerec.condition import EmbeddingBagCondition,\
    ConditionBase,\
    ConcatenationBasedConditioning, \
    ConditionalBiasing, \
    ConditionalScaling, \
    ConditionList


def test_condition_abc():

    assert issubclass(ConcatenationBasedConditioning, ConditionBase)
    assert issubclass(ConditionalBiasing, ConditionBase)
    assert issubclass(ConditionalScaling, ConditionBase)

    assert issubclass(EmbeddingBagCondition, ConditionBase)


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
