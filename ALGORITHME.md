# L'algorithme de `minvp`, en détail

Ce document décrit précisément ce que fait le paquet : les formules, les procédures,
les invariants et les garanties. Pour l'usage on se reportera au [README](README.md) ;
ici on entre dans le code (`src/`).

---

## 1. Le problème

Soit `X` une matrice de données `N × M` (N observations, M variables). On la centre
puis on normalise chaque colonne à la norme 1, ce qui donne `Z` et la **matrice de
corrélation**

```
R = ZᵀZ ∈ R^{M×M} ,   R_ii = 1.
```

Perdre une variable (l'ignorer) est une **permutation** de lignes/colonnes de `R` que
l'on garde en tête de bloc. Quand on écrit `S ⊆ {0,…,M-1}` et `R_S`, on entend la
sous-matrice principale correspondante.

```
Objectif : pour chaque taille K, trouver S de cardinal K maximisant  λ_min(R_S).
```

C'est le critère **E-optimal**. Deux reformulations utiles :

* `λ_min(R_S) = λ_min(Z_Sᵀ Z_S) = σ_min(Z_S)²` — E-optimalité = `σ_min` maximal ;
* `λ_min(R_S) = 1 / λ_max(R_S⁻¹)` — c'est le levier de l'évaluation par l'inverse (§5).

Le problème est NP-difficile. `minvp` ne cherche pas l'optimum global : il construit
une **famille imbriquée**

```
mode backward :  S_M ⊃ S_{M-1} ⊃ … ⊃ S_{k_min}
mode forward  :  S_1 ⊂ S_2 ⊂ … ⊂ S_{k_max}
```

par un glouton, mais un glouton dont **chaque étape d'élimination arrière est prouvée
optimale** (les bornes supérieures valides permettent d'élaguer sans jamais se tromper).

> Résultat : une réponse pour *toutes* les tailles K d'un seul calcul, et un drapeau
> `certified` par étape.

---

## 2. Vue d'ensemble

```
X (N×M)
  │  centres + normalisation (§3)                     matrix.rs
  ▼
Z (N×M)  ──────────────────────────────┐
  │                                     │
  │  repr = packed  (§7)                │  repr = implicit  (§7)
  ▼                                     ▼
R = ZᵀZ (packed)                        R x = Z_Sᵀ(Z_S x)
  │                                     │
  └──────────────┬──────────────────────┘
                 │
     ┌───────────┴────────────┐
     │  mode backward (§4)    │  mode forward (§6)
     │  cascade certifiée     │  glouton bordé (+ pré-filtre optionnel)
     └───────────┬────────────┘
                 │  à chaque étape : Lanczos à chaud (§8)
                 │                   inverse maintenu (§5, option)
                 │                   échanges locaux (§9, option)
                 ▼
        famille imbriquée + courbe λ_min(K) + drapeaux certified
```

Le flot `main()` (src/main.rs) est : chargement → `standardize` → `Dataset::new`
(repr) → `greedy::backward` ou `greedy::forward` → revalidation froide stricte des
sous-ensembles demandés → écriture console/CSV/JSON.

---

## 3. Prétraitement : centrage, normalisation, colonnes dégénérées

`DataMatrix::standardize` (src/matrix.rs) :

1. si `center` (défaut), retranche la moyenne de chaque colonne ;
2. calcule `‖z_j‖` et divise par la norme → colonnes de norme 1 ;
3. **élimine les colonnes de variance nulle** : seuil
   `√N · ε_machine · 100`. Les indices supprimés sont conservés et remappés à la fin
   (`keep_map`) pour que les sous-ensembles rendus parlent la numérotation d'origine.

Tout se fait par colonnes-major, en parallèle (`rayon`). Après cette étape,
`R = ZᵀZ` est bien la matrice de corrélation (ou des cosinus si `--no-center`).

---

## 4. Élimination arrière : la cascade certifiée

C'est le cœur du paquet (`greedy::backward`, `step_eliminate`, src/greedy.rs).

### 4.1 Notations de l'étape

À une étape donnée, l'ensemble courant `S` a `k` variables et l'on dispose de ses `p`
plus petits couples propres (`p = --low-rank`, défaut 4) :

```
R u_l = λ_l u_l ,   λ_1 ≤ λ_2 ≤ … ≤ λ_p ,   ‖u_l‖ = 1.
```

Retirer la variable `i` donne `R_{-i}`. Par **entrelacement de Cauchy** :

```
λ_1(R) ≤ λ_1(R_{-i}) ≤ λ_2(R) ≤ λ_2(R_{-i}) ≤ …
```

donc seule λ_1 peut augmenter, et elle reste dans `[λ_1, λ_2]`.

### 4.2 L'équation séculaire *exacte*

Les valeurs propres de `R_{-i}` sont les racines de

```
S_i(μ) = Σ_{l=1..k} u_l(i)² / (λ_l − μ) = 0.
```

*Justification* : `det(R_{-i} − μI) = det(R − μI) · [(R−μI)⁻¹]_ii` et
`[(R−μI)⁻¹]_ii = Σ_l u_l(i)²/(λ_l − μ)`. Les pôles `λ_l` étant exclus, les zéros
restants sont ceux de `S_i`.

Sur `(λ_1, λ_2)`, `S_i` est **strictement croissante**, de `−∞` (en `λ_1⁺`) à `+∞`
(en `λ_2⁻`) : elle y admet donc une **racine unique**, qui est exactement
`λ_min(R_{-i})`. Si `u_1(i) = 0`, alors `λ_min(R_{-i}) = λ_1` (le vecteur `u_1`
restreint est vecteur propre) et `secular_root` renvoie `None`.

`secular_root` (src/greedy.rs) résout cette équation par **80 bissections** sur
`(λ_1, λ_2)` (ou `(λ_1, λ_1+1)` si `p = 1`). Elle n'est utilisée que pour
**fabriquer la graine** de Lanczos, pas pour décider : c'est un majorant, pas une
valeur exacte.

Le **vecteur propre exact** de `R_{-i}` s'obtient par

```
x = (R − μI)⁻¹ e_i = Σ_l (u_l(i)/(λ_l − μ)) u_l ,     x_i = 1.
```

C'est `trial_vector` : restreint à `{0..k}\{i}`, c'est le vecteur propre exact de
`R_{-i}` évalué à la vraie racine `μ`. C'est une graine quasi parfaite pour Lanczos.

### 4.3 Borne 1 : Rayleigh (O(1) par candidat)

Pour **tout** vecteur unitaire `u`, on prive `u` de sa composante `i` et on
renormalise. Le quotient de Rayleigh de ce vecteur d'essai **majore**
`λ_min(R_{-i})` (le min sur tout l'espace est plus petit que la valeur en un point) :

```
ρ_i = ( rq − 2·u_i·y_i + u_i² ) / (1 − u_i²)  ≥  λ_min(R_{-i}),
     avec y = R u  et  rq = uᵀ R u.
```

La formule utilise `R_ii = 1`. Elle est `O(1)` mais **lâche** : elle atteint
`λ_1 + m_i` avec `m_i = u_1(i)²` (masse du vecteur propre), donc elle n'élague
presque rien. D'où la borne suivante.

### 4.4 Borne 2 : Temple spectrale p-dimensionnelle

On reprend `S_i(μ) = 0` à la racine `μ*` et on isole le terme dominant :

```
m_i / (μ* − λ_1) = Σ_{l=2..p} u_l(i)²/(λ_l − μ*) + queue(μ*).
```

On minore chacun des deux membres de droite.

**Terme basse dimension.** Puisque `μ* ≥ λ_1`, `λ_l − μ* ≤ λ_l − λ_1`, donc

```
Σ_{l=2..p} u_l(i)²/(λ_l − μ*)  ≥  B_i := Σ_{l=2..p} u_l(i)²/(λ_l − λ_1).
```

**Queue, par Cauchy–Schwarz.** Avec

```
T_i = 1 − Σ_{l≤p} u_l(i)²,        Sig_i = (1 − λ_1) − Σ_{l=2..p} u_l(i)²(λ_l − λ_1),
```

on a l'identité exacte `Sig_i = Σ_{l>p} u_l(i)²(λ_l − λ_1)` (elle utilise
`Σ_l u_l(i)² = 1` et `Σ_l u_l(i)²λ_l = R_ii = 1`), et l'inégalité de Cauchy–Schwarz
« Titu » appliquée à `a_l = u_l(i)²`, `b_l = λ_l − μ*` donne

```
queue(μ*) ≥ T_i² / (Sig_i − (μ*−λ_1)T_i) ≥ T_i² / Sig_i.
```

**Conclusion.**

```
G_i = B_i + T_i²/Sig_i,        ub_i = λ_1 + m_i / G_i  ≥  λ_min(R_{-i}).
```

C'est `LowSpectrum::upper_bound`. Pour `p = 1` elle redonne Rayleigh. Le test
`temple_bound_is_valid_and_tighter` vérifie sur des cas denses que `ub_i` domine
toujours l'exact et que l'écart moyen est plus petit que celui de Rayleigh.

**Marge pour les résidus.** Les résidus de Lanczos `‖R u_l − λ_l u_l‖` sont retranchés
de `T_i` et ajoutés à `Sig_i` :

```
slack = 2·p·max_l ‖R u_l − λ_l u_l‖
t     = max(0, 1 − Σ_{l≤p} u_l(i)² − slack)
Sig   = (1 − λ_1) − Σ_{l=2..p} u_l(i)²(λ_l − λ_1) − slack·(1 + (λ_p − λ_1))
```

Les deux opérations **agrandissent** la borne (elles diminuent `G_i`), donc restent
conservatrices. Cas particulier `t ≤ 10⁻¹²` (typiquement `p = k`, pas de queue) : le
terme de queue est supprimé et l'on prend `λ_1 + m_i/B_i`.

Enfin `upper_bound` renvoie `min(ub_i(Temple), ρ_i)`, puis borne le résultat
inférieurement par `λ_1` (garde-fou numérique).

### 4.5 Sélection et certification

Une étape = **cette boucle** :

```
1. y = R u_1 ;  rq = u_1ᵀ y                                  (1 matvec)
2. pour i = 0..k-1 :  ray_i = rayleigh(rq, u_1[i], y[i])
                      ub_i  = min(ub_i(Temple), ray_i)       O(p·k)
3. trier les candidats par borne DÉCROISSANTE
4. évaluer exactement par LOTS de `batch` (défaut 8), dans cet ordre :
       - si  meilleur_valeur ≥ borne_du_candidat_suivant − marge :  STOP, certified
       - si  max_exact atteint : STOP
5. renvoyer l'argmax exact et son couple propre
```

**Pourquoi c'est correct.** Toutes les `ub_i` sont des majorants valides de
`λ_min(R_{-i})`. Trier par borne décroissante puis évaluer exactement : dès que la
meilleure valeur **réalisée** dépasse la plus grande borne **restante**, aucun candidat
non évalué ne peut faire mieux. L'étape est alors optimale.

La marge est `tol · 50 · (1 + |rq|)`. Les ex-æquo sont tranchés par une tolérance
`tol · 10 · (1 + |valeur|)` de façon à ne pas laisser du bruit numérique décider entre
candidats indiscernables.

`StepRecord` conserve pour la traçabilité : la borne retenue (`upper`), la borne de
Rayleigh seule (`rayleigh`), le nombre de candidats évalués exactement
(`exact_evals`), le nombre total de candidats, le drapeau `certified`, le résidu de
Lanczos et les itérations cumulées.

En mode arrière **sans plafond** (`--max-exact 0`), la boucle va nécessairement au
bout ou s'arrête sur le critère : `certified = true` à chaque étape.

### 4.6 Mise à jour de l'état

Après avoir choisi `i*` :

1. `ds.swap(i*, k−1)` — la variable sort du bloc de tête ; l'ensemble vivant passe à
   `k−1` (aucune recompaction, voir §7) ; la permutation est appliquée aussi à `Z`, à
   `R` packed et à l'inverse `W` (§5) ;
2. graines pour le nouveau spectre : chaque ancien vecteur propre est privé de sa
   composante `i*` puis permuté (`restrict_and_permute`) ; `seeds[0]` est remplacé par
   le vecteur propre exact du gagnant (bien meilleur) ;
3. `low_spectrum(k−1, p, seeds, known = gagnant, cold = false)` recalcule les `p`
   plus petits couples propres **à chaud** ;
4. si `verify` : un Lanczos **froid strict** (`tol 1e-13`, 4000 itérations) recalcule
   `λ_min` et alimente `lambda_verified` — c'est la valeur affichée dans la courbe.

Point important : même quand l'évaluation exacte des candidats passe par l'inverse
(§5), le **spectre bas utilisé par les bornes est calculé en mode direct**, pour que
les bornes soient identiques à `--eval direct`.

---

## 5. Évaluation par l'inverse maintenu (`--eval inverse`)

### 5.1 Idée

`λ_min(A) = 1/λ_max(A⁻¹)` : la plus **grande** valeur propre de `A⁻¹` vaut
`1/λ_min(A)`. Inverser **amplifie** les écarts relatifs du bas du spectre :

```
{0.0050, 0.0057, 0.0061, …}   ── inverse ──▶   {200, 175, 164, …}
 écarts ~12 %
```

Lanczos converge donc en **quelques itérations** sur `A⁻¹` là où il en fallait 60 à
200 sur `A` — c'est l'effet « shift-invert », **sans factorisation par candidat**.

### 5.2 Mise à jour rang 1 (pas de factorisation par candidat)

Soit `W = R⁻¹` et `A = R_{-i}` le bloc privé de `i`. En partitionnant
`R = [[A, b],[bᵀ, R_ii]]`, l'identité d'inverse par blocs donne

```
A⁻¹ = W_{-i,-i} − (1/W_ii) w wᵀ ,      w = W_{·,i} privé de i.
```

Trois opérations, chacune `O(k²)` ou `O(k)` :

| opération | code | coût |
|---|---|---|
| Cholesky initial (`W = R⁻¹`) | `InverseSym::from_packed` | `O(M³)` une fois |
| permutation `i ↔ j` du bloc de tête | `swap_leading` | `O(k)` |
| suppression du dernier indice | `downdate_last` | `O(k²)` |
| matvec `W_{-d,-d} x − (wᵀx)w/W_dd` | `mul` | `O(k²)` |

Le matvec `mul` scinde ses produits scalaires en **deux segments contigus** de part
et d'autre de l'indice supprimé `d`, au lieu de tester `u < d` à chaque élément :
la boucle `O(k²)` devient vectorisable.

`from_packed` renvoie `None` si le bloc n'est pas défini positif (`M > N` singulier) ;
on retombe alors sur l'évaluation directe. `downdate_error_growth` vérifie que
`W R − I` reste sous `10⁻⁶` après des centaines de mises à jour.

### 5.3 L'opérateur `−A⁻¹`

`NegInvSubOp` applique `x ↦ −A⁻¹ x` (via `mul(..., negate = true)`). Sa **plus
petite** valeur propre `θ_min` vaut `−1/λ_min(A)`, d'où la conversion

```
λ_min(R_{-i}) = −1 / θ_min(−A⁻¹).
```

`to_lambda` implémente cette conversion. Le test `inverse_eval_matches_direct_eval`
vérifie que les deux voies donnent les mêmes valeurs et vecteurs à la tolérance près.

---

## 6. Sélection avant : le glouton bordé

`greedy::forward` part d'un singleton et ajoute une variable par étape.

### 6.1 Première variable

`best_first_feature` :

* si `R` est matérialisée (`packed`) : l'indice dont la corrélation maximale en valeur
  absolue avec les autres est la plus faible — heuristique de type Gershgorin,
  `O(M²)` (utilisée jusqu'à `M ≤ 20 000`) ;
* sinon (implicite) : la colonne la plus éloignée du centroïde, `O(NM)`.

### 6.2 Classement par perte séculaire bordée

La matrice bordée `B = [[R_S, c],[cᵀ, 1]]` (`c = Z_Sᵀ z_j`) a pour valeurs propres
propres celles de `R_S` **plus** les racines de

```
1 − μ = Σ_l g_l² / (λ_l − μ),      g_l = u_lᵀ c.
```

La **perte au premier ordre** en ajoutant `j` vaut `Σ_l g_l²/(λ_l − λ_1)` : chaque
direction propre contribue d'autant plus qu'elle est proche de `λ_1`. C'est le score
`loss` de `forward`, et c'est ce qui distingue `minvp` du filtre par corrélation, qui
ne regarde que `|g_1|`.

Le calcul des `g_{l,·}` pour **toutes** les colonnes se fait par deux passes
vectorisées et parallèles : `w_l = Z_S u_l` (`z_loading`), puis
`g_{l,j} = z_j · w_l` pour tout `j` (`z_dot_all`, `O(NM)` par couple propre, en
parallèle sur les colonnes).

### 6.3 Pré-filtre (optionnel) et évaluation exacte

```
mode exact (forward_top = 0, défaut, CERTIFIÉ)   mode pré-filtre (forward_top = N)
  ub_j  = racine séculaire tronquée  (majorant)    scored = tri par loss CROISSANTE
  tri par ub DÉCROISSANTE                          take   = min(N, M−k)
  évaluer par lots, s'arrêter dès que              évaluer exactement les `take`
  meilleur_réalisé ≥ ub_suivant − marge            premiers, prendre le meilleur
```

Aucune bascule automatique selon la taille : par défaut (`forward_top = 0`) la
sélection avant est le glouton E-optimal **certifié**, au sens où chaque étape
s'arrête sur le critère d'optimalité (drapeau `certified`) au lieu d'évaluer les
`M−k` candidats. Le pré-filtre s'active explicitement — `prefilter=true` dans le
binding Python (largeur 16, ou `forward_top=N`), `--prefilter` / `--forward-top N`
en ligne de commande ; il n'évalue alors que les `N` meilleurs du score séculaire et
perd la certification.

Chaque candidat retenu est évalué exactement par Lanczos à chaud sur `S ∪ {j}` :

* si `R` est **matérialisée**, l'opérateur est `BorderedHeadOp`, qui applique
  directement `[[R_S, c],[cᵀ, 1]]` : `c` est la ligne `j` du triangle packed, lue de
  façon **contiguë** (`j ≥ k`), et le matvec coûte `O(k²)` ;
* sinon (`implicite`) on retombe sur `GatheredZOp`, qui construit
  `R_{S∪{j}} x = Cᵀ(Cx)` avec `C = [Z_S, z_j]`, soit `O(Nk)` par matvec.

Le second est le facteur limitant du glouton avant exact dès que `N` est grand
(`2N ≳ k`), d'où le premier.

La graine est le **vecteur d'essai KKT**

```
mu  = racine de  1 − μ − Σ_l g_l²/(λ_l − μ) = 0   (racine sous λ_1 ; majorant si tronquée)
y   = −Σ_l (g_l/(λ_l − mu)) u_l ,     graine = [ y ; 1 ],
```

repli sur `[u_1 ; 1]` si la racine n'est pas exploitable ou si `‖y‖ > 10⁶`.

**Conséquence importante sur la garantie** : avec `forward_top = 0` (défaut), chaque
étape est certifiée optimale (mêmes majorants que la cascade arrière, §4.4) — mais
sans évaluer les `M−k` candidats. Avec le pré-filtre activé, il n'évalue que les `N`
meilleurs du score séculaire : c'est une **heuristique**, plus rapide (coût par étape
~constant au lieu de croître avec `M`) mais dont le `λ_min` peut être inférieur de
quelques pour cent (§4 du README) — benchmarké dans `python/bench_numpy.py`.

### 6.4 État après ajout

`ds.swap(k, j)` place la variable retenue en position `k` ; les vecteurs propres
précédents sont étendus d'un `0` et servent de graines à chaud ; le gagnant alimente
`known`.

---

## 7. Représentations et coûts

Deux représentations, choisies par `Repr::Auto` (`Dataset::new`) :

| | `packed` | `implicit` |
|---|---|---|
| stockage | triangle inférieur de `R`, `M(M+1)/2` f64 | `Z` seul (`N·M`) |
| matvec `R_S x` | `O(k²)` (`matvec_head`) | `O(Nk)` : `Z_Sᵀ(Z_S x)` |
| retenu par `Auto` si | tient dans `--mem-budget` | sinon |
| avantage théorique | `k` petit | `2N ≲ k` |
| construction | `O(NM²)`, GEMM bloqué | — |

Dans les deux cas **l'ensemble actif est toujours le bloc de tête** `0..k`. Supprimer
une variable n'est qu'un échange `i ↔ k−1` :

* `Z` : échange de deux colonnes contiguës, `O(N)` ;
* `PackedSym::swap_leading` : parcourt les lignes jusqu'à `M` pour permuter les deux
  indices, soit `O(M)` ; en contrepartie **aucune recompaction** `O(M²)` n'a lieu, la
  partie vivante restant le bloc de tête ;
* `InverseSym::swap_leading` : `O(k)` (borné au bloc de tête).

La construction de `R` est un GEMM bloqué parallèle : blocs de lignes (`--block-rows`,
défaut 64) et blocs de colonnes de 64, un bloc de lignes restant résident en cache
pendant le balayage.

Petit utilitaire numérique partagé (src/lib.rs) : produit scalaire et norme
déroulés ×4 pour favoriser la vectorisation, `axpy` avec court-circuit sur `alpha = 0`.

---

## 8. Le solveur spectral : Lanczos reorthogonalisé

`lanczos::smallest_eigenpair_constrained` (src/lanczos.rs) est le point d'entrée
unique ; il trouve la **plus petite** valeur propre d'un opérateur `SymOp`.

### 8.1 La boucle

Lanczos classique avec :

* **reorthogonalisation complète** contre toute la base (stabilité), dont la
  **seconde passe est conditionnelle** : elle n'est exécutée que si la première a fait
  chuter la norme de `w` sous `1/√2` de sa valeur d'entrée — signe d'une annulation
  catastrophique, donc d'une perte d'orthogonalité. Les itérations chaudes, où `w`
  perd peu, économisent ainsi une passe complète sur la base ;
* **projection explicite** contre les vecteurs de contrainte `constraints` à chaque
  étape (déflation, §8.2) ;
* **démarrage à chaud** : la graine est le vecteur propre de l'étape précédente, ou le
  vecteur KKT séculaire ;
* **test de convergence sur le résidu de Ritz** : à chaque itération `j`, la plus
  petite valeur propre `θ_j` de la tridiagonale `T_j` est estimée par
  `tridiag_smallest` (bissection + suite de Sturm, 40 pas), puis le résidu
  `‖A u − θ_j u‖ = β_j·|y_j|` (`y_j` = dernière composante du vecteur propre de `T_j`,
  obtenu par itération inverse, `O(j)`) est comparé à `tol·(1 + |θ_j|)`. Un simple test
  de stabilisation `|θ_j − θ_{j−1}|` peut se déclencher sur un palier alors que le
  résidu est encore grand : la valeur de Ritz — un **majorant** de `λ_min` — est alors
  surestimée, ce qui fausse la certification de l'étape gloutonne ;
* **détection de sous-espace invariant** : si `β ≤ 10⁻⁹·(1 + |α|)`, `w` n'est plus que
  du bruit d'arrondi ; on s'arrête avant de créer des valeurs propres parasites.

**Fin.** La plus petite valeur propre de `T_m` est recalculée par bissection (60 pas),
son vecteur propre par itération inverse (`tridiag_smallest_eigenvector`, `O(m)` — au
lieu de la diagonalisation de Jacobi `O(m³)` qui coûtait plus cher que tout le reste
de la boucle chaude) ; le **résidu exact** `‖A u − θ u‖` est calculé sur le vecteur
final (indicateur de qualité du *vecteur*, pas de la valeur).

### 8.2 Pourquoi la déflation est « explicite »

Pour trouver le 2ᵉ, 3ᵉ… couple propre, `low_spectrum_inv` rappelle Lanczos en lui
interdisant l'espace déjà couvert. Deux implémentations possibles :

* éclater sur l'opérateur projeté `(I−P)A` : le vecteur d'essai conserve une
  composante parasite qui se fait amplifier par `α/β` quand `β` est petit → instable ;
* **retenue ici** : orthogonaliser chaque nouveau vecteur de base contre les
  contraintes. C'est stable par construction.

### 8.3 Solveurs denses auxiliaires (src/jacobi.rs)

* `eigen_sym` : Jacobi cyclique, `O(n³)`, jusqu'à 100 balayages, arrêt quand
  `Σ_{p<q} a_pq² ≤ 10⁻³⁰ · Σ a_ii²`. Très robuste, réservé aux tests et aux petites
  tailles ;
* `tridiag_smallest` : bissection sur `[−r−1, r+1]` où `r` est le rayon de Gershgorin
  de la tridiagonale ; le test d'intervalle n'a besoin que du prédicat « au moins une
  valeur propre sous `μ` », d'où `sturm_has_negative` (suite de Sturm avec **sortie
  anticipée** dès le premier pivot négatif — le compteur de Sturm est croissant) ;
* `tridiag_smallest_eigenvector` / `tridiag_inverse_iterate` : vecteur propre de la
  plus petite valeur propre par itération inverse (LU de `T − θI` factorisée une fois,
  `O(n)` par itération) ;
* `tridiag_rayleigh` : quotient de Rayleigh `yᵀTy`, `O(n)`.

Le tableau des itérations Lanczos n'utilise **jamais** de re-diagonalisation `O(j³)`
dans la boucle chaude.

---

## 9. Échanges locaux 1-contre-1 (`--swap-passes`)

Amélioration locale à taille `K` constante (`refine_swaps`), activée si
`--swap-passes n > 0` et seulement pour `k ≤ --swap-from` :

```
pour chaque passe :
  ρ_i = borne de Rayleigh du retrait de i ∈ S          (candidats sortants)
  g_j = u_1ᵀ c_j = w · z_j   pour j ∉ S,  w = Z_S u_1   (candidats entrants)
  former les `swap-top × swap-top` paires (sortant i, entrant j),
  les scorer par ρ_i − |g_j| croissant, garder les `batch` meilleures,
  évaluer exactement chacune (GatheredZOp avec del = i, extra = j),
  accepter la meilleure si la valeur progresse strictement, sinon STOP.
```

C'est une montée de gradient discrète, **monotone par construction** (on n'accepte
qu'une amélioration réelle) mais non certifiée.

---

## 10. Complexité

Notations : `k` = taille courante, `e` = candidats évalués exactement par étape,
`t` = itérations de Lanczos par candidat.

**Backward** (M → k_min), par étape :

| poste | coût |
|---|---|
| borne de Rayleigh | 1 matvec + `O(k)` |
| bornes de Temple | `O(p·k)` |
| évaluations exactes | `e · t ·` matvec |
| spectre bas à chaud | `p` Lanczos contraints |
| permutation `i* ↔ k−1` | `O(M)` (packed) + `O(N)` (`Z`) |
| (inverse) mise à jour `W` | `O(k²)` |

avec matvec `O(k²)` (packed) ou `O(Nk)` (implicite). Le point clé est que **`e` reste
petit** : le tri par borne décroissante fait que la certification arrive souvent après
un ou deux candidats, au lieu des `k` d'une recherche exhaustive.

**Forward** (1 → k_max), par étape : `O(p·NM)` pour tous les scores, `O(M log M)` de
tri, `O(M·p·J)` pour les bornes séculaires (`J` = pas de bissection de
`addition_secular`), puis `e · t ·` matvec pour les évaluations. Depuis la
certification avant, **`e` reste petit** comme en arrière : `take = M−k` n'est plus
le comportement par défaut, le tri par borne séculaire décroissante permet de
s'arrêter dès que la meilleure valeur réalisée dépasse la plus grande borne restante.
Avec le pré-filtre (`forward_top = N > 0`), `e = N` mais sans garantie.

**Empreinte mémoire.** `packed` : `M(M+1)/2` f64 + `W` (`M²`) si l'inverse est actif ;
`implicit` : `N·M` seulement.

---

## 11. Garanties, et ce qui les limite

**Garanti.**

* Les bornes de Rayleigh et de Temple sont des **majorants valides** de
  `λ_min(R_{-i})` ; les résidus sont intégrés de façon conservatrice. Tests :
  `rayleigh_upper_is_valid_for_any_unit_vector`,
  `temple_bound_is_valid_and_tighter`.
* La racine séculaire tronquée de la matrice bordée est un majorant valide de la
  nouvelle `λ_min`, ce qui certifie la sélection avant exacte. Test :
  `forward_secular_bound_quality` (validité + écart mesuré à l'exact).
* En mode arrière sans plafond de candidats, chaque étape s'arrête sur le critère de
  certification : le retrait choisi est celui du glouton exact. Tests :
  `backward_matches_brute_force_greedy` (comparaison à la force brute par `eigvalsh`).
* `forward` avec `forward_top = 0` réalise le même glouton exact, mais certifié et
  économe. Test : `forward_matches_brute_force_greedy`.
* `implicit` et `packed` donnent les mêmes résultats ; inverse et direct aussi. Tests :
  `backward_implicit_matches_packed`, `inverse_eval_matches_direct_eval`.

**Non garanti / limites.**

* Le glouton est **myope** : la famille imbriquée n'est pas globalement optimale
  (les échanges locaux du §9 corrigent partiellement).
* `--eval direct --iters-warm 60` : les évaluations peuvent ne pas être convergées, la
  certification est alors à prendre avec la marge de tolérance ; `--eval inverse` ou un
  `--iters-warm` plus grand règlent le problème. Le solveur s'arrête désormais sur le
  **résidu** de Ritz, ce qui rend cette limite visible (`converged = false`) plutôt que
  silencieuse.
* `M > N` : `R` est singulière, pas d'inverse (`from_packed` → `None`), on retombe sur
  les évaluations directes.
* `forward` avec le pré-filtre activé (`--prefilter`, `--forward-top N > 0`,
  `prefilter=true` côté Python) : heuristique, `λ_min` non garanti (cf. §6.3).
* Quand deux candidats ont des `λ_min` à moins de la marge de certification
  (`tol·50·(1+|λ|)`) l'un de l'autre, le choix est arbitraire : sur une famille
  longue, cette indétermination peut faire diverger la suite des sous-ensembles. Les
  étapes restent optimales *à la marge près* (vérifié pas à pas contre la force brute
  sur données i.i.d. et corrélées).

---

## 12. Paramètres qui changent le comportement

| option | effet | défaut |
|---|---|---|
| `--dir backward\|forward` | sens du parcours | `backward` |
| `--kmin K` / `--kmax K` | bornes du parcours | `1` / `M` |
| `--tol F` | tolérance Lanczos, marge de certification | `1e-10` |
| `--iters-cold N` | itérations à froid | `400` |
| `--iters-warm N` | itérations à chaud | `60` |
| `--low-rank P` | nombre `p` de couples propres pour Temple | `4` |
| `--max-exact N` | plafond de candidats évalués par étape (`0` = certifié) | `0` |
| `--eval auto\|direct\|inverse` | voie d'évaluation | `auto` |
| `--forward-top N` | largeur du pré-filtre avant (`0` = désactivé, glouton exact certifié) | `0` |
| `--prefilter` | active le pré-filtre avant (largeur 16) | désactivé |
| `--forward-first J` | première variable imposée | heuristique |
| `--swap-passes N`, `--swap-top N`, `--swap-from K` | échanges locaux | `0`, `8`, `kmin` |
| `--repr auto\|packed\|implicit` | représentation | `auto` |
| `--mem-budget MB`, `--block-rows N`, `--threads N` | performance | `4096`, `64`, tous |
| `--no-verify` | désactive la revalidation froide | vérifie |

`auto` pour `--eval` choisit l'inverse si `R` est matérialisée et `M ≤ 3000`
(`inverse_max_m`).

---

## 13. Sorties et traçabilité

* **courbe** `(K, λ_min)` : `lambda_verified` quand la revalidation froide est active,
  sinon la valeur de Ritz à chaud ;
* **par étape** : variable entrée/sortie, borne supérieure retenue, borne de Rayleigh,
  nombre de candidats évalués, drapeau `certified`, résidu, itérations ;
* **`--out-csv`** : colonnes
  `k,lambda_min_ritz,lambda_min_verifie,variable_changee,borne_sup,exact_evals,candidats,certifie,residu,iterations` ;
* **`--out-json`** : `meta`, `direction`, `initial_k`, `initial_lambda`, `seconds`,
  `total_exact_evals`, `total_lanczos_iters`, `certified_steps`, `order`,
  `initial_subset`, `curve`, `subsets`.

Avant de rendre un sous-ensemble, `main.rs` le **revalide** : il replace les variables
choisies en tête et relance un Lanczos **froid** très strict (`tol 1e-13`, 4000
itérations). La valeur affichée est donc une valeur de Ritz certifiée à la tolérance
près, indépendamment de tout démarrage à chaud.

`order` et `initial_subset` suffisent à reconstruire n'importe quel `S_K` de la famille
(`Selection::subset_at` du binding Python).

---

## 14. Où lire quoi

| fichier | contenu |
|---|---|
| `src/greedy.rs` | cascade, bornes, certification, glouton avant, échanges |
| `src/lanczos.rs` | Lanczos reorthogonalisé + déflation + arrêt Sturm |
| `src/jacobi.rs` | Jacobi dense, bissection tridiagonale, suite de Sturm |
| `src/inverse.rs` | inverse maintenu, `−A⁻¹`, mise à jour rang 1 |
| `src/op.rs` | `SymOp`, `Dataset`, `SubOp`, `GatheredZOp`, `z_loading`/`z_dot_all` |
| `src/packed.rs` | triangle inférieur packed, GEMM bloqué, matvec, permutations |
| `src/matrix.rs` | standardisation, colonnes dégénérées |
| `src/gen.rs` | générateurs synthétiques, RNG xorshift64* |
| `src/io.rs` | texte/CSV/binaire/stdin f64 |
| `src/report.rs` | console, CSV, JSON |
| `src/cli.rs` | analyse des arguments |
| `crates/minvp-python/src/lib.rs` | extension PyO3 : `run`, `Selection`, `Step`, GIL, threads |
| `python/minvp/__init__.py` | API Python (`select`/`path`/`curve`), conversion des entrées |
| `python/test_minvp.py` | tests du binding (numpy comme oracle) |
| `python/bench_numpy.py` | comparaison de temps vs glouton naïf numpy |
| `python/compare_baseline.py` | qualité vs heuristiques de corrélation |
