"""Type stubs for the ``sifters`` package."""

from typing import Any, Callable, Optional

from ._sifters import SiftersError as SiftersError
from ._sifters import Selection as Selection
from ._sifters import Step as Step

__version__: str
__all__: list[str]

ProgressCallback = Callable[[Step], Optional[bool]]

def version() -> str: ...
def build_profile() -> str: ...
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
    forward_seeds: int = ...,
    exact: bool = ...,
    exact_time_s: float = ...,
    exact_max_evals: int = ...,
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
    forward_seeds: int = ...,
    swap_passes: int = ...,
    swap_top: int = ...,
    swap_from: Optional[int] = ...,
    mem_budget_mb: int = ...,
    block_rows: int = ...,
    threads: Optional[int] = ...,
    progress: Optional[ProgressCallback] = ...,
) -> Selection: ...

def curve(X: Any, **kwargs: Any) -> dict[int, float]: ...

# -- nonlinear / non-Gaussian dependence (see sifters.dependence, sifters.dopt) --
def dependence_matrix(X: Any, *, method: str = ..., **kwargs: Any) -> Any: ...
def select_from_dependence(
    D: Any, k: int, *, criterion: str = ..., **kwargs: Any
) -> Any: ...
def select_dependence(
    X: Any,
    k: int,
    *,
    method: str = ...,
    criterion: str = ...,
    measure_kwargs: Optional[dict] = ...,
    **kwargs: Any,
) -> Any: ...
