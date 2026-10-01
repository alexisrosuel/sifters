"""minvp — sélection E-optimale de variables, en Rust (binding natif PyO3).

**`minvp` choisit, pour chaque taille K, le sous-ensemble de K variables dont la
matrice de corrélation est la mieux conditionnée** : celui qui maximise sa plus
petite valeur propre ``lambda_min(R_S)`` (critère *E-optimal*).

Le cœur numérique est écrit en Rust sûr et multithread ; ce module est une
extension native : aucune donnée ne transite par un sous-processus, les tableaux
numpy sont lus via le protocole tampon, et le GIL est libéré pendant le calcul.

Exemple
-------
>>> import numpy as np, minvp
>>> X = np.random.default_rng(0).normal(size=(500, 300))   # (observations, variables)
>>> r = minvp.select(X, k=30)
>>> r.subset[:5]
[3, 7, 11, 29, 41]
>>> r.lambda_min
0.123...
>>> r.curve[10]          # lambda_min de la famille pour K = 10
0.29...

La famille est **imbriquée** : ``r.subset_at(k)`` donne le sous-ensemble de
n'importe quelle taille ``k`` visitée, sans recalcul (mode ``"backward"``) ou
après complétion (mode ``"forward"``).
"""

from __future__ import annotations

import array
import os
from typing import Any, Callable, Optional

from . import _minvp
from ._minvp import MinvpError, Selection, Step, __version__

__all__ = [
    "select",
    "path",
    "curve",
    "as_matrix",
    "Selection",
    "Step",
    "MinvpError",
    "version",
    "__version__",
]

#: Signature d'un callback de progression : reçoit un :class:`Step` et renvoie
#: ``False`` (ou un objet faux) pour interrompre le parcours.  ``None`` poursuit.
ProgressCallback = Callable[[Step], Optional[bool]]

REMOVED = {
    "binary_path": (
        "binary_path() a été supprimé : le paquet n'appelle plus l'exécutable "
        "`minvp` par sous-processus, le moteur est une extension native."
    ),
    "Result": (
        "Result a été remplacé par minvp.Selection (mêmes attributs principaux : "
        "k, subset, lambda_min, curve, seconds, order, initial_subset, direction, "
        "m_total, subset_at())."
    ),
}


def __getattr__(name: str) -> Any:
    if name in REMOVED:
        raise AttributeError(REMOVED[name])
    raise AttributeError(f"module 'minvp' n'a pas d'attribut '{name}'")


def version() -> str:
    """Version du moteur (identique à ``minvp.__version__``)."""
    return __version__


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------

def _as_buffer(X: Any) -> tuple[Any, Optional[tuple[int, int]], int]:
    """Convertit ``X`` en tampon exploitable et renvoie ``(tampon, forme, M)``.

    numpy est utilisé s'il est présent (il absorbe listes, DataFrames, tenseurs,
    ``float32``, vues non contiguës...).  Sans numpy, ``X`` doit déjà exposer un
    tampon ``float64`` 2D, ou être une séquence de séquences (aplatie ici).
    """
    if isinstance(X, (str, bytes, os.PathLike)):
        raise TypeError(
            "passer un chemin de fichier n'est pas supporté : chargez les données "
            "vous-même (numpy.loadtxt, pandas.read_csv, ...) puis passez le tableau"
        )
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        a = np.asarray(X)
        if a.ndim != 2:
            raise ValueError(
                f"X doit être 2D (observations, variables), reçu {a.ndim}D"
            )
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a, None, int(a.shape[1])

    if isinstance(X, memoryview) and X.ndim == 2:
        return X, None, int(X.shape[1])

    # Repli sans numpy : séquence de séquences -> tampon 1D (lignes-major).
    try:
        rows = [list(r) for r in X]  # type: ignore[union-attr]
    except TypeError:
        return X, None, 0
    if not rows:
        raise ValueError("X est vide")
    n, m = len(rows), len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("toutes les lignes de X doivent avoir la même longueur")
    flat = array.array("d", [float(v) for r in rows for v in r])
    return flat, (n, m), m


def as_matrix(X: Any) -> Any:
    """Renvoie une vue 2D ``float64`` contiguë (ordre C) de ``X``.

    Utile pour vérifier ou préparer explicitement une entrée ; les fonctions de
    sélection font déjà cette conversion.  numpy est utilisé s'il est installé.
    """
    buf, shape, _ = _as_buffer(X)
    if shape is not None:  # repli sans numpy : on renvoie le tampon aplati
        return buf
    try:
        import numpy as np

        return np.asarray(buf, dtype=np.float64)
    except ImportError:  # pragma: no cover - dépend de l'environnement
        return buf


def _auto_direction(m: int, k: int) -> str:
    """``forward`` est bien meilleur (et bien moins cher) que ``backward`` dès
    que l'on cherche un K petit devant M."""
    if m <= 0:
        return "forward"
    return "forward" if k * 3 < m else "backward"


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def select(
    X: Any,
    k: int,
    *,
    direction: str = "auto",
    center: bool = True,
    verify: bool = True,
    eval: str = "auto",
    representation: str = "auto",
    low_rank: int = 4,
    tol: float = 1e-10,
    iters_warm: int = 60,
    iters_cold: int = 400,
    max_exact: int = 0,
    batch: int = 8,
    prefilter: bool = False,
    forward_top: int = 0,
    forward_first: Optional[int] = None,
    swap_passes: int = 0,
    swap_top: int = 8,
    swap_from: Optional[int] = None,
    mem_budget_mb: int = 4096,
    block_rows: int = 64,
    threads: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> Selection:
    """Sélectionne ``k`` variables parmi ``X`` (critère E-optimal).

    Paramètres
    ----------
    X : array-like, forme ``(n_observations, n_variables)``
        Converti en ``float64`` contigu ; les colonnes sont centrées puis
        normalées (``center=False`` donne la matrice des cosinus).
    k : int
        Nombre de variables à retenir.
    direction : {"auto", "backward", "forward"}
        ``"auto"`` (défaut) choisit ``"forward"`` si ``3k < M``, sinon
        ``"backward"``.  ``"backward"`` construit la famille complète certifiée
        (coûteux en grand ``M``) ; ``"forward"`` ajoute gloutonnement.
    center : bool
        Centre les colonnes (corrélation) ou non (cosinus).
    verify : bool
        Revalide ``lambda_min`` du sous-ensemble retenu par un Lanczos froid
        strict (tolérance ``1e-13``).  En mode ``backward``, revalide aussi
        chaque étape de la courbe.
    eval : {"auto", "direct", "inverse"}
        Évaluation des candidats.  ``"inverse"`` maintient ``R⁻¹`` et converge
        en ~20 itérations au lieu de 100–150 (corrélation définie positive).
    representation : {"auto", "packed", "implicit"}
        Corrélation matérialisée (``packed``) ou appliquée implicitement
        ``Z_Sᵀ(Z_S x)`` (``implicit``, optimal si ``N ≪ k``).
    low_rank : int
        Nombre de couples propres utilisés par la borne de Temple (défaut 4).
    tol, iters_warm, iters_cold : réglages Lanczos.
    max_exact : int
        Plafond de candidats évalués exactement par étape (0 = certifié).
    forward_top : int
        En mode ``forward``, largeur du pré-filtre séculaire.  ``0`` (défaut)
        évalue tous les candidats : le glouton est alors exactement E-optimal.
        Ignoré si ``prefilter=False`` ; sinon fixe la largeur (``0`` → 16).
    prefilter : bool
        Active le pré-filtre séculaire en mode ``forward`` : seuls les
        ``forward_top`` (16 par défaut) meilleurs candidats du score séculaire
        sont évalués exactement.  **Désactivé par défaut** : le glouton avant est
        alors exactement E-optimal.  Aucune activation automatique selon la
        taille.  L'activer accélère fortement les grands ``M``, au prix de
        quelques pour cent de ``lambda_min`` (heuristique) — voir le README.
    forward_first : int, optionnel
        Première variable imposée (mode ``forward``).
    swap_passes, swap_top, swap_from : échanges locaux 1-contre-1 (mode
        ``backward``), pour corriger la myopie du glouton.
    mem_budget_mb, block_rows : performance / mémoire.
    threads : int, optionnel
        Taille du pool ``rayon`` dédié.  ``None`` (ou ``0``) utilise le pool
        global, dimensionné sur le nombre de cœurs disponibles.
    progress : callable, optionnel
        Appelé avec un :class:`Step` après chaque étape.  Renvoyer ``False``
        interrompt proprement le parcours (le résultat partiel est renvoyé) ;
        renvoyer ``None`` poursuit.  Les ``Ctrl-C`` sont interceptés au même
        moment et lèvent ``KeyboardInterrupt``.

    Retour
    ------
    :class:`Selection`

    Exemples
    --------
    >>> r = minvp.select(X, k=30)
    >>> r.subset, r.lambda_min
    ([3, 7, ...], 0.123...)
    """
    if not isinstance(k, (int,)) or isinstance(k, bool):
        try:
            k = int(k)
        except (TypeError, ValueError) as exc:  # pragma: no cover - défensif
            raise TypeError("k doit être un entier") from exc
    buf, shape, m = _as_buffer(X)
    if direction == "auto":
        direction = _auto_direction(m, int(k))
    return _minvp.run(
        buf,
        shape=shape,
        direction=direction,
        k=int(k),
        center=center,
        verify=verify,
        eval=eval,
        representation=representation,
        low_rank=low_rank,
        tol=tol,
        iters_warm=iters_warm,
        iters_cold=iters_cold,
        max_exact=max_exact,
        batch=batch,
        prefilter=prefilter,
        forward_top=forward_top,
        forward_first=forward_first,
        swap_passes=swap_passes,
        swap_top=swap_top,
        swap_from=swap_from,
        mem_budget_mb=mem_budget_mb,
        block_rows=block_rows,
        threads=threads,
        progress=progress,
    )


def path(
    X: Any,
    *,
    kmin: int = 1,
    kmax: Optional[int] = None,
    direction: str = "backward",
    center: bool = True,
    verify: bool = True,
    eval: str = "auto",
    representation: str = "auto",
    low_rank: int = 4,
    tol: float = 1e-10,
    iters_warm: int = 60,
    iters_cold: int = 400,
    max_exact: int = 0,
    batch: int = 8,
    prefilter: bool = False,
    forward_top: int = 0,
    forward_first: Optional[int] = None,
    swap_passes: int = 0,
    swap_top: int = 8,
    swap_from: Optional[int] = None,
    mem_budget_mb: int = 4096,
    block_rows: int = 64,
    threads: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> Selection:
    """Construit une famille imbriquée de sous-ensembles.

    En ``direction="backward"`` (défaut), part des ``M`` variables et élimine
    jusqu'à ``kmin`` : la famille ``S_M ⊃ … ⊃ S_kmin`` est **certifiée**
    (chaque étape est prouvée optimale).  En ``direction="forward"``, part d'un
    singleton et ajoute jusqu'à ``kmax``.

    Renvoie un :class:`Selection` dont :attr:`~Selection.curve` donne
    ``lambda_min`` pour tous les K traversés et dont
    :meth:`~Selection.subset_at` reconstruit n'importe quel sous-ensemble.

    Les paramètres sont ceux de :func:`select`.
    """
    buf, shape, _ = _as_buffer(X)
    return _minvp.run(
        buf,
        shape=shape,
        direction=direction,
        kmin=int(kmin),
        kmax=None if kmax is None else int(kmax),
        center=center,
        verify=verify,
        eval=eval,
        representation=representation,
        low_rank=low_rank,
        tol=tol,
        iters_warm=iters_warm,
        iters_cold=iters_cold,
        max_exact=max_exact,
        batch=batch,
        prefilter=prefilter,
        forward_top=forward_top,
        forward_first=forward_first,
        swap_passes=swap_passes,
        swap_top=swap_top,
        swap_from=swap_from,
        mem_budget_mb=mem_budget_mb,
        block_rows=block_rows,
        threads=threads,
        progress=progress,
    )


def curve(X: Any, **kwargs: Any) -> dict[int, float]:
    """Raccourci : renvoie directement la courbe ``{K: lambda_min}``.

    Accepte les mêmes mots-clés que :func:`path`.
    """
    return dict(path(X, **kwargs).curve)
