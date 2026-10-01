#!/usr/bin/env bash
# Comparaison A/B robuste : min sur N repetitions du temps CPU `user` et du temps
# de parcours (`seconds` du JSON).  Le temps mural est contamine par la charge de
# la machine ; `user` mesure le travail reellement effectue.
#
# Usage : bench/ab.sh <label> <ref_binary> [scenario...] [all]
#   REP=3 bench/ab.sh ...     nombre de repetitions
set -u
cd "$(dirname "$0")/.."
LABEL=${1:?label}
REF=${2:?binaire de reference}
shift 2
REP=${REP:-3}
OUT=bench/out/$LABEL
mkdir -p "$OUT"

SCENARIOS=(
  "fwd400|5,10,20,50|--gen blocks --gen-n 500 --gen-m 400 --blocks 8 --dir forward --kmax 50"
  "fwd800|5,10,20,50|--gen blocks --gen-n 500 --gen-m 800 --blocks 8 --dir forward --kmax 50"
  "fwd3200|5,10,20,50|--gen blocks --gen-n 500 --gen-m 3200 --blocks 8 --dir forward --kmax 50"
  "bwd200|5,10,20,50|--gen blocks --gen-n 600 --gen-m 200 --blocks 8 --dir backward --kmin 1"
  "bwd500i|250,300,400|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200 --eval inverse"
  "bwd500d|250,300,400|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200 --eval direct"
  "bwd500c|250,300,400|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200 --eval direct --iters-warm 300"
  "bwd500def|250,300,400|--gen blocks --gen-n 600 --gen-m 500 --blocks 8 --dir backward --kmin 200"
)

if [ $# -gt 0 ] && [ "$1" != "all" ]; then
  filtered=()
  for spec in "${SCENARIOS[@]}"; do
    for want in "$@"; do [ "${want%%|*}" = "${spec%%|*}" ] && filtered+=("$spec"); done
  done
  SCENARIOS=("${filtered[@]}")
fi

printf '%-11s %9s %9s %7s  %-16s %s\n' scenario ref_cpu now_cpu cpu_x "exacts" "parcours ref->now"
for spec in "${SCENARIOS[@]}"; do
  name=${spec%%|*}; rest=${spec#*|}; subset=${rest%%|*}; args=${rest#*|}
  ref_cpu=999999; now_cpu=999999; ref_path=999999; now_path=999999
  for tag in ref now; do
    if [ "$tag" = ref ]; then bin=$REF; else bin=${MINVP_BIN:-./target/release/minvp}; fi
    best_cpu=999999; best_path=999999
    for _ in $(seq "$REP"); do
      # shellcheck disable=SC2086
      /usr/bin/time -p -o /tmp/minvp_time.txt $bin $args --quiet \
          --out-json "$OUT/${name}_$tag.json" --subset "$subset" \
          >"$OUT/${name}_$tag.log" 2>&1
      cpu=$(awk '/^user/{print $2}' /tmp/minvp_time.txt)
      path=$(python3 -c "import json;print(json.load(open('$OUT/${name}_$tag.json'))['seconds'])" 2>/dev/null || echo 999)
      best_cpu=$(python3 -c "print(min($best_cpu, ${cpu:-999999}))")
      best_path=$(python3 -c "print(min($best_path, ${path:-999999}))")
    done
    if [ "$tag" = ref ]; then ref_cpu=$best_cpu; ref_path=$best_path; else now_cpu=$best_cpu; now_path=$best_path; fi
  done
  python3 - "$OUT/$name" "$ref_cpu" "$now_cpu" "$ref_path" "$now_path" <<'PY'
import json,sys
b,rc,nc,rp,np_=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])
r=json.load(open(b+'_ref.json')); n=json.load(open(b+'_now.json'))
same = 'OK' if r['order']==n['order'] else 'ORDER-DIFFERS'
print('%-11s %9.2f %9.2f %6.2fx  %-16s %.2f -> %.2f s  %s' % (
   b.split('/')[-1], rc, nc, (rc/nc if nc else 0),
   '%d->%d'%(r['total_exact_evals'],n['total_exact_evals']), rp, np_, same))
PY
done
