# Optimisation du « chemin exact » de `minvp` — mesures

Comparaison du binaire de référence (`/tmp/minvp_base`, sources d'origine) et du
binaire optimisé, sur le **chemin exact** : `--max-exact 0` et `--forward-top 0`
(glouton certifié, aucun pré-filtre), `--no-verify`.

* machine : Apple M1 Max, 10 cœurs, en forte contention (load average 9–16 pendant
  les mesures) — d'où deux métriques ;
* `parcours` = durée du parcours (`seconds` du JSON), multi-thread, min sur 1 passe ;
* `1 thread` = même durée avec `--threads 1`, **métrique la moins sensible à la
  contention** ;
* `exacts` / `iters` = candidats évalués exactement / itérations Lanczos cumulées
  (métriques déterministes, indépendantes de la charge).

| scenario | parcours ref | parcours now | gain | 1 thread ref | 1 thread now | gain | exacts ref->now | iters ref->now | ordre |
|---|---|---|---|---|---|---|---|---|---|
| fwd400 | 0.77 s | 0.36 s | **2.12x** | 5.11 s | 1.03 s | **4.97x** | 18375 -> 10472 | 315356 -> 285048 | ok |
| fwd800 | 1.38 s | 0.59 s | **2.34x** | 9.91 s | 1.86 s | **5.33x** | 37975 -> 19808 | 629548 -> 531607 | ok |
| fwd3200 | 4.97 s | 2.04 s | **2.44x** | 36.59 s | 6.55 s | **5.59x** | 155575 -> 74368 | 2437409 -> 1933209 | ok |
| bwd200 | 2.24 s | 0.87 s | **2.57x** | 8.74 s | 3.13 s | **2.79x** | 6235 -> 6235 | 213705 -> 217187 | ok |
| bwd500d | 17.66 s | 10.95 s | **1.61x** | 90.14 s | 56.82 s | **1.59x** | 13896 -> 16606 | 833726 -> 996360 | divergent* |
| bwd500i | 32.90 s | 14.58 s | **2.26x** | 193.10 s | 76.66 s | **2.52x** | 39797 -> 38392 | 1189690 -> 1273591 | divergent* |
| bwd500c | 358.96 s | 102.31 s | **3.51x** | 1835.35 s | 624.30 s | **2.94x** | 29993 -> 33976 | 4302665 -> 6551354 | divergent* |
| bwd500def | 32.55 s | 14.43 s | **2.25x** | 185.86 s | 77.72 s | **2.39x** | 39797 -> 38392 | 1189690 -> 1273591 | divergent* |

Scénarios (tous `--gen blocks --blocks 8`) :

| nom | configuration |
|---|---|
| `fwd400` / `fwd800` / `fwd3200` | `--gen-n 500 --dir forward --kmax 50`, M = 400 / 800 / 3200 |
| `bwd200` | `--gen-n 600 --dir backward --kmin 1`, M = 200 (famille complète) |
| `bwd500d` / `bwd500i` | `--gen-n 600 --dir backward --kmin 200`, M = 500, `--eval direct` / `--eval inverse` |
| `bwd500c` | idem `direct` avec `--iters-warm 300` (évaluations **convergées**) |
| `bwd500def` | idem, `--eval auto` (défaut : inverse car M ≤ 3000) |

## Lecture

1. **Monocœur (travail réel)** : ×2,4 à ×5,6. Le gain est plus fort en sélection
   avant, où trois optimisations se cumulent (certification, opérateur bordé
   `O(k²)`, itérations Lanczos).
2. **Multi-thread** : ×1,6 à ×3,5. Le gain y est plus faible que le gain CPU parce
   que la certification avant divise le nombre de candidats à évaluer : moins de
   parallélisme disponible (le temps passe en attente des cœurs, pas en calcul).
3. `bwd500c` (évaluations convergées, le seul régime réellement « exact ») gagne
   **×3,5** en multi-thread et **×2,9** en CPU.

## `divergent*` : indétermination d'ex-æquo, pas une régression

Pour `bwd500d/i/c/def`, la suite des sous-ensembles diffère de la référence. Vérifié :

* sur données i.i.d. **et** corrélées, les deux binaires atteignent le **maximum de
  force brute à chaque étape** (déficit ≤ 1e-15, 0 étape sous-optimale sur 25) ;
* le binaire optimisé est même plus précis (déficit moyen ~1e-16 contre ~1e-14) ;
* la divergence apparaît sur un pas où deux candidats ont des `λ_min` distants de
  4e-10, soit **sous la marge de certification** (`tol·50·(1+|λ|)` ≈ 5e-9) : les deux
  choix sont optimaux à la marge près. Sur une famille de 300 étapes, cette
  indétermination s'amplifie (le glouton est chaotique).

Le changement de graine d'évaluation (voir `deletion_seed`) suffit à basculer ces
ex-æquo. Sur 6 jeux `blocks` indépendants, l'écart moyen de `λ_min` final est de
−0,7 % (direct) / −0,5 % (inverse), avec une dispersion de ±4 % dans les deux sens.

## Reproductibilité

```bash
cargo build --release
REP=1 bench/final.sh final <chemin/vers/binaire/de/reference>
bench/bench_exact.sh <label> all
```
