"""Cache-folder resolution for transitlos (never CWD-relative)."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

CACHE_ENV_VAR = "TRANSITLOS_CACHE_DIR"


def get_cache_dir(
    namespace: str = "",
    cache_dir: Optional[Union[str, Path]] = None,
    create: bool = True,
) -> Path:
    """Resolve a cache folder: explicit ``cache_dir`` (returned as is), else
    ``$TRANSITLOS_CACHE_DIR/<namespace>``, else ``<tempdir>/transitlos-cache/<namespace>``.

    Args:
        namespace: Sub-folder of the cache root.
        cache_dir: Explicit folder; returned as is.
        create: Create the folder if missing.

    Returns:
        The cache folder.
    """
    if cache_dir:
        path = Path(cache_dir)
    else:
        env = os.environ.get(CACHE_ENV_VAR)
        root = Path(env).expanduser() if env else Path(tempfile.gettempdir()) / "transitlos-cache"
        path = root / namespace if namespace else root
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_osm_dir(cache_dir: Optional[Union[str, Path]] = None, create: bool = True) -> Path:
    """Folder for ``.osm.pbf`` street extracts (namespace ``osm``); see :func:`get_cache_dir`.

    Args:
        cache_dir: Explicit folder; returned as is.
        create: Create the folder if missing.

    Returns:
        The OSM cache folder.
    """
    return get_cache_dir("osm", cache_dir, create)
