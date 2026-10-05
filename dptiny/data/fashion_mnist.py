"""Fashion-MNIST dataset loader.

Same interface as :func:`dptiny.data.mnist.get_mnist`, so training code can
swap one call for the other. The four gzipped IDX files are downloaded from
the project's GitHub on first use, cached, and parsed with NumPy only.
"""

import gzip
import math
import os
import tempfile
import urllib.request
from typing import Optional, Tuple

import numpy as np

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")

_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
    "master/data/fashion/"
)
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

#: Class names in label order (label 0 is "T-shirt/top", ..., 9 is "Ankle boot").
CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

_IDX_UINT8 = 0x08


def _download(url: str, path: str, retries: int = 3) -> str:
    """Download ``url`` to ``path`` unless it is already cached.

    The data is written to a temporary file in the same directory. It is only
    renamed to ``path`` (an atomic operation) after the download is complete
    AND the file has been checked: its size must match the server's
    Content-Length, and it must decompress and parse as a valid IDX file.
    A connection can end early without raising an error, so this check is
    what guarantees the cache never holds a half-written file.
    """
    if os.path.exists(path):
        return path

    directory = os.path.dirname(path)
    for attempt in range(1, retries + 1):
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url) as response:
                expected = response.headers.get("Content-Length")
                written = 0
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
            if expected is not None and written != int(expected):
                raise ValueError(
                    f"incomplete download of {url}: got {written} of "
                    f"{expected} bytes"
                )
            _read_idx_gz(tmp_path)  # raises ValueError if the file is broken
            os.replace(tmp_path, path)  # atomic rename
            return path
        except (OSError, ValueError) as e:
            _remove(tmp_path)
            if attempt == retries:
                raise
            print(f"Download failed ({e}); retrying ({attempt}/{retries})...")
        except BaseException:  # e.g. KeyboardInterrupt: clean up, then stop
            _remove(tmp_path)
            raise
    return path


def _remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def parse_idx(data: bytes) -> np.ndarray:
    """Parse the raw bytes of a uint8 IDX file into a NumPy array.

    Header layout: two zero bytes, a data-type code (0x08 = uint8), the number
    of dimensions, then each dimension size as a big-endian 32-bit integer.

    Raises:
        ValueError: if the data is not a uint8 IDX file, or if its size does
            not match the dimensions in its header.
    """
    if len(data) < 4:
        raise ValueError("not an IDX file: too short to contain a header")
    if data[0] != 0 or data[1] != 0:
        raise ValueError("not an IDX file: the first two bytes must be zero")
    if data[2] != _IDX_UINT8:
        raise ValueError(
            f"not a uint8 IDX file: data-type code is 0x{data[2]:02x}, "
            f"expected 0x{_IDX_UINT8:02x}"
        )
    ndim = data[3]
    if ndim == 0:
        raise ValueError("invalid IDX file: number of dimensions is 0")

    header_len = 4 + 4 * ndim
    if len(data) < header_len:
        raise ValueError("invalid IDX file: header is truncated")

    dims = tuple(
        int(d) for d in np.frombuffer(data, dtype=">u4", count=ndim, offset=4)
    )
    expected = header_len + math.prod(dims)
    if len(data) != expected:
        raise ValueError(
            f"IDX file size mismatch: header {dims} implies {expected} bytes, "
            f"file has {len(data)}"
        )

    return np.frombuffer(data, dtype=np.uint8, offset=header_len).reshape(dims)


def _read_idx_gz(path: str) -> np.ndarray:
    """Decompress a gzipped IDX file and parse it."""
    try:
        with gzip.open(path, "rb") as f:
            data = f.read()
    except (OSError, EOFError) as e:
        raise ValueError(f"cannot decompress {path}: {e}") from e
    return parse_idx(data)


def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Fashion-MNIST dataset.

    Args:
        normalize: If ``True``, scale pixel values to ``[0, 1]``.
        flatten: If ``True``, return images as ``(N, 784)`` vectors,
            otherwise as ``(N, 1, 28, 28)``.
        data_home: Cache directory (default ``~/.cache/dptiny``). The files
            are stored in its ``fashion_mnist`` subfolder.

    Returns:
        ``(X_train, X_test, y_train, y_test)``: float32 images and int32
        labels, with 60,000 training and 10,000 test examples.
    """
    cache_dir = os.path.join(
        data_home if data_home is not None else _CACHE_DIR, "fashion_mnist"
    )
    os.makedirs(cache_dir, exist_ok=True)

    arrays = {}
    for key, filename in _FILES.items():
        url = _BASE_URL + filename
        path = _download(url, os.path.join(cache_dir, filename))
        try:
            arrays[key] = _read_idx_gz(path)
        except ValueError:
            # A damaged file in the cache (e.g. left by another tool):
            # remove it and download a fresh, verified copy once.
            _remove(path)
            arrays[key] = _read_idx_gz(_download(url, path))

    X_train, y_train = arrays["train_images"], arrays["train_labels"]
    X_test, y_test = arrays["test_images"], arrays["test_labels"]

    for X, y, name in ((X_train, y_train, "train"), (X_test, y_test, "test")):
        if X.ndim != 3 or X.shape[1:] != (28, 28):
            raise ValueError(f"unexpected {name} image shape {X.shape}")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"{name} labels {y.shape} do not match images {X.shape}")

    # Same steps, in the same order, as get_mnist.
    X_train = X_train.reshape(len(X_train), 784).astype(np.float32)
    X_test = X_test.reshape(len(X_test), 784).astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    if normalize:
        X_train = X_train / np.float32(255.0)
        X_test = X_test / np.float32(255.0)

    if not flatten:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test
