"""Shared pytest fixtures. Several test files take a `tmp: Path` argument so they
also run as plain scripts (see their __main__ blocks); map it to pytest's tmp_path."""

import pytest


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


@pytest.fixture
def v(tmp_path):
    """test_validation.py's CardValidator, built the way its __main__ block builds it."""
    from test_validation import make_validator
    return make_validator(tmp_path)
