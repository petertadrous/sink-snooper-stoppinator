import pytest
from src.deterrent._deterrent import Deterrent


class DummyDeterrent(Deterrent):
    def setup(self):
        return "setup"

    def activate(self, duration: float):
        return f"activate {duration}"

    def cleanup(self):
        return "cleanup"


def test_abstract_methods_enforced():
    with pytest.raises(TypeError):
        Deterrent()


def test_dummy_deterrent_methods():
    d = DummyDeterrent()
    assert d.setup() == "setup"
    assert d.activate(1.0) == "activate 1.0"
    assert d.cleanup() == "cleanup"
