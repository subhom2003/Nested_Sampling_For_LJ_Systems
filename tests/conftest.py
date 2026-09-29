"""Shared test fixtures and utilities."""
import tempfile
import pytest

from ljns.config import tiny_test_cfg


@pytest.fixture
def tmp_outdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def cfg(tmp_outdir):
    """Tiny validated config for fast unit tests (N=8, K=4, n_iter=40)."""
    return tiny_test_cfg(tmp_outdir)