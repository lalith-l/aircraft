"""
Pytest fixtures shared across all tests.
"""
import pytest
import random
import numpy as np


@pytest.fixture
def seed():
    """Fixed seed for reproducible tests. All test runs must be deterministic."""
    s = 42
    random.seed(s)
    np.random.seed(s)
    return s


@pytest.fixture
def seed_alt():
    """Alternative seed for multi-seed comparisons."""
    s = 7
    random.seed(s)
    np.random.seed(s)
    return s
