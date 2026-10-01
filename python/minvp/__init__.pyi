"""Stubs du paquet ``minvp``."""

from typing import Any, Callable, Optional

from ._minvp import MinvpError as MinvpError
from ._minvp import Selection as Selection
from ._minvp import Step as Step

__version__: str
__all__: list[str]

ProgressCallback = Callable[[Step], Optional[bool]]

def version() -> str: ...
def as_matrix(X: Any) -> Any: ...

def select(
    X: Any,
    k: int,
    *,
    direction: str = ...,
    center: bool = ...,
    verify: bool = ...,
    eval: str = ...,
    representation: str = ...,
    low_rank: int = ...,
    tol: float = ...,
    iters_warm: int = ...,
    iters_cold: int = ...,
    max_exact: int = ...,
    batch: int = ...,
    prefilter: bool = ...,
    forward_top: int = ...,
    forward_first: Optional[int] = ...,
    swap_passes: int = ...,
    swap_top: int = ...,
    swap_from: Optional[int] = ...,
    mem_budget_mb: int = ...,
    block_rows: int = ...,
    threads: Optional[int] = ...,
    progress: Optional[ProgressCallback] = ...,
) -> Selection: ...

def path(
    X: Any,
    *,
    kmin: int = ...,
    kmax: Optional[int] = ...,
    direction: str = ...,
    center: bool = ...,
    verify: bool = ...,
    eval: str = ...,
    representation: str = ...,
    low_rank: int = ...,
    tol: float = ...,
    iters_warm: int = ...,
    iters_cold: int = ...,
    max_exact: int = ...,
    batch: int = ...,
    prefilter: bool = ...,
    forward_top: int = ...,
    forward_first: Optional[int] = ...,
    swap_passes: int = ...,
    swap_top: int = ...,
    swap_from: Optional[int] = ...,
    mem_budget_mb: int = ...,
    block_rows: int = ...,
    threads: Optional[int] = ...,
    progress: Optional[ProgressCallback] = ...,
) -> Selection: ...

def curve(X: Any, **kwargs: Any) -> dict[int, float]: ...
