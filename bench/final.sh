#!/usr/bin/env bash
# Benchmark final A/B du « chemin exact » : reference vs binaire courant.
#
#   bench/final.sh <label> <ref_binary>
#
# Mesure, pour chaque scenario :
#   * `parcours` : duree du parcours (champ `seconds` du JSON), multi-thread, min de REP ;
#   * `1 thread` : meme duree avec --threads 1 (metrique peu sensible a la contention) ;
#   * `exacts`   : candidats evalues exactement ; `iters` : iterations Lanczos.
set -u
cd "$(dirname "$0")/.."
LABEL=${1:?label}
REF=${2:?binaire de reference}
REP=${REP:-2}
OUT=bench/out/$LABEL
mkdir -p "$OUT"

SCENARIOS=(
  "fwd400|--gen blocks --gen-n 500 --gen-m 400 --blocks 8 --dir forward --kmax 50"
  "fwd800|--gen blocks --gen-n 500 --gen-m 800 --blocks 8 --dir forward --kmax 50"
  "fwd3200|--gen blocks --gen-n 500 --gen-m 3200 --blocks 8 --dir forward --kmax 50"
  "fwd1600k150|--gen blocks --gen-n 500 --gen-m 1600 --blocks 8 --dir forward --kmax 150"
  "bwd200|--gen blocks --gen-n 600 --gen-m 200 --blocks 8 --dir backward --kmin 1"
  "bwd500d|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200 --eval direct"
  "bwd500i|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200 --eval inverse"
  "bwd500c|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200 --eval direct --iters-warm 300"
  "bwd500def|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200"
)

run_one() { # bin args threads out
  local bin=$1 args=$2 th=$3 out=$4 best=999999
  for _ in $(seq "$REP"); do
    # shellcheck disable=SC2086
    $bin $args --quiet --no-verify --threads "$th" --out-json "$out" >/dev/null 2>&1
    local p
    p=$(python3 -c "import json;print(json.load(open('$out'))['seconds'])" 2>/dev/null || echo 999999)
    best=$(python3 -c "print(min($best, ${p:-999999}))")
  done
  echo "$best"
}

printf '| scenario | parcours ref | parcours now | gain | 1 thread ref | 1 thread now | gain | exacts ref->now | iters ref->now | ordre |\n'
printf '|---|---|---|---|---|---|---|---|---|---|\n'
for spec in "${SCENARIOS[@]}"; do
  name=${spec%%|*}; args=${spec#*|}
  rp=$(run_one "$REF" "$args" 10 "$OUT/${name}_ref.json")
  np=$(run_one "${MINVP_BIN:-./target/release/minvp}" "$args" 10 "$OUT/${name}_now.json")
  r1=$(run_one "$REF" "$args" 1  "$OUT/${name}_ref1.json")
  n1=$(run_one "${MINVP_BIN:-./target/release/minvp}" "$args" 1 "$OUT/${name}_now1.json")
  python3 - "$OUT/$name" "$rp" "$np" "$r1" "$n1" <<'PY'
import json,sys
b,rp,np_,r1,n1 = sys.argv[1], *map(float, sys.argv[2:6])
r=json.load(open(b+'_ref.json')); n=json.load(open(b+'_now.json'))
same='ok' if r['order']==n['order'] else 'divergent*'
print('| %s | %.2f s | %.2f s | **%.2fx** | %.2f s | %.2f s | **%.2fx** | %d -> %d | %d -> %d | %s |' % (
  b.split('/')[-1], rp, np_, (rp/np_ if np_ else 0), r1, n1, (r1/n1 if n1 else 0),
  r['total_exact_evals'], n['total_exact_evals'],
  r['total_lanczos_iters'], n['total_lanczos_iters'], same))
PY
done
