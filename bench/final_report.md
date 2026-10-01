# Optimisation du « chemin exact » de `minvp` — mesures

Comparaison du binaire de **référence** (sources d'origine, `/tmp/minvp_base`) et du
binaire optimisé, sur le **chemin exact** : `--max-exact 0` et `--forward-top 0`
(glouton certifié, aucun pré-filtre), `--no-verify`.

* machine : Apple M1 Max, 10 cœurs, en forte contention (load average 9–16 pendant
  les mesures) — d'où deux métriques ;
* `parcours` = durée du parcours (`seconds` du JSON), multi-thread, `REP=1` ;
* `1 thread` = même durée avec `--threads 1`, **métrique la moins sensible à la
  contention** ;
* `exacts` / `iters` = candidats évalués exactement / itérations Lanczos cumulées
  (métriques déterministes, indépendantes de la charge).

## Sélection avant (`forward --forward-top 0`)

| scénario | parcours ref | parcours now | gain | 1 thread ref | 1 thread now | gain | exacts ref->now | iters ref->now | ordre |
|---|---|---|---|---|---|---|---|---|---|
| fwd400 (M=400, K=50) | 0.92 s | 0.22 s | **4.10x** | 5.11 s | 0.21 s | **23.8x** | 18375 -> 392 | 315356 -> 392 | ok |
| fwd800 (M=800, K=50) | 1.62 s | 0.29 s | **5.52x** | 9.98 s | 0.36 s | **28.1x** | 37975 -> 392 | 629548 -> 392 | ok |
| fwd3200 (M=3200, K=50) | 5.30 s | 0.75 s | **7.04x** | 36.84 s | 1.22 s | **30.1x** | 155575 -> 392 | 2437409 -> 392 | ok |
| fwd1600k150 (M=1600, K=150) | 37.91 s | 8.40 s | **4.51x** | 265.67 s | 10.34 s | **25.7x** | 227225 -> 1192 | 6547307 -> 32701 | ok |

Les sous-ensembles sont **identiques** à la référence dans les quatre cas. Mieux : les
`λ_min` rapportées y sont **exactes**, car la graine KKT construite sur le spectre
complet converge en une poignée d'itérations. La référence, elle, surestime (ses
évaluations à chaud saturent `--iters-warm 60`) : sur `fwd1600k150` à K=112,
0,334289554 (référence) contre 0,334279286 (optimisé) — la revalidation froide stricte
donne 0,334279286397, résidu 1e-13. Idem à K=150 : 0,254933341 (optimisé) contre
0,254933379 (référence) pour une valeur exacte de 0,254933340508.

## Élimination arrière (`backward --max-exact 0`)

| scénario | parcours ref | parcours now | gain | 1 thread ref | 1 thread now | gain | exacts ref->now |
|---|---|---|---|---|---|---|---|
| bwd200 (M=200, famille complète) | 2.23 s | 0.85 s | **2.62x** | 8.75 s | 3.16 s | **2.77x** | 6235 -> 6235 |
| bwd500d (M=500, kmin=200, `direct`) | 17.69 s | 11.04 s | **1.60x** | 89.45 s | 56.49 s | **1.58x** | 13896 -> 16606 |
| bwd500i (idem, `inverse`) | 32.90 s | 14.58 s | **2.26x** | 193.10 s | 76.66 s | **2.52x** | 39797 -> 38392 |
| bwd500c (idem, `direct` **convergé**, `--iters-warm 300`) | 358.96 s | 102.31 s | **3.51x** | 1835.35 s | 624.30 s | **2.94x** | 29993 -> 33976 |
| bwd500def (idem, `--eval auto`, le défaut) | 32.55 s | 14.43 s | **2.25x** | 185.86 s | 77.72 s | **2.39x** | 39797 -> 38392 |

Scénarios : `--gen blocks --blocks 8`, `--gen-n 500` (avant) ou `600` (arrière).

## Lecture

1. **Sélection avant : ×23 à ×30 en travail réel.** Trois effets se cumulent —
   certification (plus d'évaluation exhaustive), opérateur bordé `O(k²)` sur la
   corrélation matérialisée, et surtout borne séculaire **exacte** (`p = k`) qui
   réduit les évaluations au plancher `batch` (8 par étape) : les itérations Lanczos
   passent de 2,4 M à 392 sur `fwd3200` (~6000× moins).
2. **Élimination arrière : ×1,6 à ×3,5.** Le gain vient de la graine séculaire enfin
   utilisée (~30 % d'itérations en moins par candidat), du vecteur de Ritz par
   itération inverse, de la seconde passe de reorthogonalisation conditionnelle et du
   matvec de l'inverse sans branche.
3. **Multi-thread < mono-thread.** La certification réduit le nombre de candidats à
   évaluer : il reste peu de parallélisme, et le temps part en synchronisation. Le
   gain « travail » (mono-cœur) est la mesure honnête.

## `divergent*` : indétermination d'ex-æquo, pas une régression

Pour certains scénarios arrière, la suite des sous-ensembles diffère de la
référence. Vérifié :

* sur données i.i.d. **et** corrélées, les deux binaires atteignent le **maximum de
  force brute à chaque étape** (déficit ≤ 1e-15, 0 étape sous-optimale sur 25) ;
  c'est également vrai du chemin avant (test `forward_matches_brute_force_greedy`,
  et `test_forward_exact_is_stepwise_optimal` côté Python) ;
* le binaire optimisé est même plus précis (déficit moyen ~1e-16 contre ~1e-14) ;
* la divergence apparaît sur un pas où deux candidats ont des `λ_min` distants de
  4e-10, soit **sous la marge de certification** (`tol·50·(1+|λ|)` ≈ 5e-9) : les deux
  choix sont optimaux à la marge près. Sur une famille de 300 étapes, cette
  indétermination s'amplifie (le glouton est chaotique).

Sur 6 jeux `blocks` indépendants, l'écart moyen de `λ_min` final est de −0,7 %
(direct) / −0,5 % (inverse), avec une dispersion de ±4 % dans les deux sens.

## Bug pré-existant corrigé au passage : la revalidation CLI

`minvp --subset K` revalide le sous-ensemble retenu par un Lanczos froid strict. La
revalidation construisait un `Dataset` neuf — dont `active` repart à `0..m` — à partir
de `ds.z`, **déjà permuté** par le parcours : la correspondance position → indice
d'origine était donc rompue et la revalidation portait sur un autre sous-ensemble
(0,6036 au lieu de 0,8634 sur un test arbitré par numpy). Corrigé par
`head_subset_dataset`, avec le test de régression
`head_subset_dataset_validates_the_right_subset`.

## Reproductibilité

```bash
cargo build --release
REP=1 bench/final.sh final <chemin/vers/binaire/de/reference> [scenario...]
bench/bench_exact.sh <label> all
```
