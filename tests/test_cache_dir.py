import tempfile
from pathlib import Path

from transitlos.cache import CACHE_ENV_VAR, get_cache_dir


def test_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    assert get_cache_dir("g", create=False) == Path(tempfile.gettempdir()) / "transitlos-cache" / "g"
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
    assert get_cache_dir("g") == tmp_path / "g" and (tmp_path / "g").is_dir()
    assert get_cache_dir("g", tmp_path / "x", create=False) == tmp_path / "x"
