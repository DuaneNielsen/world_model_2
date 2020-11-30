import pytest
from pytest import fixture
from worldmodel import upper_tri, v_nk, v_l
import torch

r0, r1, r2, r3 = 0.1, 0.2, 0.3, 0.4
v0, v1, v2, v3 = 0.5, 0.6, 0.7, 0.8


@fixture
def rstack():
    r = torch.tensor([r0, r1, r2, r3])
    rstack = torch.stack([r, r], dim=1).unsqueeze(2)
    return rstack


@fixture
def vstack():
    v = torch.tensor([v0, v1, v2, v3])
    vstack = torch.stack([v, v], dim=1).unsqueeze(2)
    return vstack


@fixture
def triangle():
    triangle = torch.empty(3, 4, 2)
    element = torch.tensor([
        [r0, v1, 0, 0],
        [r0, r1, v2, 0],
        [r0, r1, r2, v3],
    ])

    triangle[:, :, 0] = element
    triangle[:, :, 1] = element
    triangle = triangle.unsqueeze(-1)
    return triangle


def test_upper_tri(rstack, vstack, triangle):
    upper = upper_tri(rstack, vstack)
    assert torch.allclose(triangle, upper)


def test_vnk(triangle):
    d = 0.9
    VNK = v_nk(triangle, d)
    element = torch.tensor([
        [r0, v1, 0, 0],
        [r0, r1, v2, 0],
        [r0, r1, r2, v3],
    ])

    discount = torch.tensor([
        [d ** 0 / 2, d ** 1 / 2, 0, 0],
        [d ** 0 / 3, d ** 1 / 3, d ** 2 / 3, 0],
        [d ** 0 / 4, d ** 1 / 4, d ** 2 / 4, d ** 3 / 4],
    ])
    element *= discount
    element = element.sum(1)
    element = torch.stack([element, element], dim=1).unsqueeze(2)
    assert torch.allclose(VNK, element)


def test_v_l_with_wunsie():
    wunsie = torch.ones(3, 2, 1)
    VL = v_l(wunsie, lam=0.9)
    assert torch.allclose(VL, torch.ones(2, 1))


def test_vl_with_example():
    VNK = torch.tensor([[[0.3200],
                         [0.3200]],

                        [[0.2823],
                         [0.2823]],

                        [[0.2765],
                         [0.2765]]])

    lam = 0.9
    expected_VL = (1 - lam) * (0.3200 * lam ** 0 + 0.2823 * lam ** 1) + (0.2765 * lam ** 2)
    expected_VL = torch.tensor([expected_VL, expected_VL]).unsqueeze(1)
    VL = v_l(VNK, lam=lam)
    assert torch.allclose(expected_VL, VL)