"""Block ProPainter's automatic weight download. Startup already checked local files."""

from pathlib import Path
from urllib.parse import urlparse


def local_only(url, model_dir=None, progress=True, file_name=None, **kwargs):
    target = Path(model_dir or "weights") / (file_name or Path(urlparse(url).path).name)
    if not target.is_file():
        raise RuntimeError("offline model asset missing")
    return str(target)


def install_offline_hook():
    try:
        from utils import download_util
    except Exception:
        return False
    download_util.load_file_from_url = local_only
    return True
