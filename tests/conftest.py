import pytest

from reslot.generate import generate
from reslot.replay import run_replay


@pytest.fixture(scope="session")
def ds():
    return generate()


@pytest.fixture(scope="session")
def result(ds):
    return run_replay(ds)
