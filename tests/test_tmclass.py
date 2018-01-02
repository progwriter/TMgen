# coding: utf-8
import pytest
from tmgen import TrafficMatrix


def test_empty_constructor():
    with pytest.raises(TypeError):
        TrafficMatrix()
