"""Type stubs for the native extension module ``sifters._sifters``."""

from typing import Any, Callable, Optional

__version__: str


class SiftersError(RuntimeError): ...

class Step:
    k_before: int
    k_after: int
    index: int
    lambda_min: float
    lambda_verified: float
    upper_bound: float
    rayleigh_bound: float
    exact_evals: int
    candidates: int
    certified: bool
    residual: float
    lanczos_iters: int
    def __repr__(self) -> str: ...

class Selection:
    direction: str
    k: int
    subset: list[int]
    lambda_min: float
    lambda_residual: float
    curve: dict[int, float]
    seconds: float
    path_seconds: float
    load_seconds: float
    order: list[int]
    initial_subset: list[int]
    n_observations: int
    m_total: int
    m_usable: int
    dropped: list[int]
    steps: list[Step]
    certified_steps: int
    total_exact_evals: int
    total_lanczos_iters: int
    representation: str
    eval: str
    proved_optimal: bool
    gap_certified: float
    exact_evals: int
    def subset_at(self, k: int) -> Optional[list[int]]: ...
    def lambda_at(self, k: int) -> Optional[float]: ...
    def lambda_raw_at(self, k: int) -> Optional[float]: ...
    def to_dict(self) -> dict[str, Any]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

def build_profile() -> str:
    """Compilation profile of the extension: ``"release"`` or ``"debug"``."""
    ...

def run(
    x: Any,
    *,
    shape: Optional[tuple[int, int]] = ...,
    direction: str = ...,
    k: Optional[int] = ...,
    kmin: Optional[int] = ...,
    kmax: Optional[int] = ...,
    verify: bool = ...,
    center: bool = ...,
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
    eval: str = ...,
    representation: str = ...,
    mem_budget_mb: int = ...,
    block_rows: int = ...,
    threads: Optional[int] = ...,
    progress: Optional[Callable[[Step], Optional[bool]]] = ...,
) -> Selection: ...
