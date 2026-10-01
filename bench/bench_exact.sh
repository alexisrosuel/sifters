#!/usr/bin/env bash
# Benchmark du « chemin exact » de minvp (aucun pre-filtre, max_exact = 0).
#
# Usage :
#   bench/bench_exact.sh <label> [scenario...]
#   bench/bench_exact.sh <label> all
#
# Un scenario s'ecrit  "nom|liste_de_K|arguments".
# Chaque scenario ecrit un JSON + CSV dans bench/out/<label>/ ; les JSON servent
# de reference de non-regression : une optimisation valide doit rendre les memes
# sous-ensembles et la meme courbe.
set -u
cd "$(dirname "$0")/.."
BIN=${MINVP_BIN:-./target/release/minvp}
LABEL=${1:-run}
shift || true
OUT=bench/out/$LABEL
mkdir -p "$OUT"

SCENARIOS=(
  "fwd400|5,10,20,50|--gen blocks --gen-n 500 --gen-m 400 --blocks 8 --dir forward --kmax 50"
  "fwd800|5,10,20,50|--gen blocks --gen-n 500 --gen-m 800 --blocks 8 --dir forward --kmax 50"
  "fwd3200|5,10,20,50|--gen blocks --gen-n 500 --gen-m 3200 --blocks 8 --dir forward --kmax 50"
  "bwd200|5,10,20,50|--gen blocks --gen-n 600 --gen-m 200 --blocks 8 --dir backward --kmin 1"
  "bwd500i|250,300,400|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200 --eval inverse"
  "bwd500d|250,300,400|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200 --eval direct"
)

if [ $# -gt 0 ] && [ "$1" != "all" ]; then
  filtered=()
  for spec in "${SCENARIOS[@]}"; do
    name=${spec%%|*}
    for want in "$@"; do
      [ "${want%%|*}" = "$name" ] && filtered+=("$spec")
    done
  done
  SCENARIOS=("${filtered[@]}")
fi

for spec in "${SCENARIOS[@]}"; do
  name=${spec%%|*}
  rest=${spec#*|}
  subset=${rest%%|*}
  args=${rest#*|}
  t0=$(date +%s.%N)
  # shellcheck disable=SC2086
  $BIN $args --quiet --no-verify --out-json "$OUT/$name.json" \
      --subset "$subset" --out-csv "$OUT/$name.csv" >"$OUT/$name.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  printf '%-10s %8.2f s (rc=%d)\n' "$name" "$(echo "$t1 - $t0" | bc)" "$rc"
done
