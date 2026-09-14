"""Process-local, pickle-safe read-only NumPy memmaps.

``numpy.memmap`` inherits NumPy's ndarray pickling behavior: pickling an open
memmap serializes its complete payload.  PyTorch's ``forkserver`` and ``spawn``
DataLoader workers pickle their Dataset, so keeping large memmaps directly on
the Dataset can materialize hundreds of GiB before the first batch.

``LazyMemmap`` pickles only the file specification and opens the mapping on
first access in each process.  It intentionally implements just the small
array surface used by Mellow's cache readers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


class LazyMemmap:
    """A pickle-safe proxy that lazily opens one NumPy memmap per process."""

    def __init__(
        self,
        path: str | Path,
        *,
        dtype: Any,
        mode: str = "r",
        shape: tuple[int, ...],
        offset: int = 0,
        order: str = "C",
    ) -> None:
        if mode != "r":
            raise ValueError("LazyMemmap is intentionally read-only")
        self.path = str(Path(path).resolve())
        self._dtype_string = np.dtype(dtype).str
        self.mode = mode
        self.shape = tuple(int(value) for value in shape)
        self.offset = int(offset)
        self.order = str(order)
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError(f"invalid LazyMemmap shape: {self.shape}")
        self._array: np.memmap | None = None
        self._owner_pid: int | None = None

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._dtype_string)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def filename(self) -> str:
        return self.path

    def _open(self) -> np.memmap:
        pid = os.getpid()
        if self._array is None or self._owner_pid != pid:
            self._array = np.memmap(
                self.path,
                dtype=self.dtype,
                mode=self.mode,
                shape=self.shape,
                offset=self.offset,
                order=self.order,
            )
            self._owner_pid = pid
        return self._array

    def __getitem__(self, key: Any) -> Any:
        return self._open()[key]

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        array = np.asarray(self._open(), dtype=dtype)
        if copy:
            return array.copy()
        return array

    def __getstate__(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "dtype": self._dtype_string,
            "mode": self.mode,
            "shape": self.shape,
            "offset": self.offset,
            "order": self.order,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.path = str(state["path"])
        self._dtype_string = str(state["dtype"])
        self.mode = str(state["mode"])
        self.shape = tuple(int(value) for value in state["shape"])
        self.offset = int(state["offset"])
        self.order = str(state["order"])
        self._array = None
        self._owner_pid = None

    def __repr__(self) -> str:
        return (
            f"LazyMemmap(path={self.path!r}, dtype={self.dtype!r}, "
            f"shape={self.shape!r})"
        )
