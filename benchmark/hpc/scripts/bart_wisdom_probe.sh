#!/bin/bash
# Does BART_USE_FFTW_WISDOM=1 actually speed up BART's *timed* runs (run 1 discarded, as the
# comparison suite does)? Run on an exclusive/quiet node — the login node's process pressure
# makes BART abort and the numbers meaningless.
#
#   TOOLBOX_PATH=/project/c_mrrecon/bart_openblas benchmark/hpc/scripts/bart_wisdom_probe.sh
set -u
BART="${TOOLBOX_PATH:?set TOOLBOX_PATH to the bart binary}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT; cd "$WORK"
export BART_FFTW_WISDOM_DIR="$WORK/wisdom"

# representative problem: 2x-undersampled 128x128 8-coil TV pics, same shape as the suite
"$BART" phantom -x128 -s8 img_coil
"$BART" fft -u 7 img_coil ksp_full
"$BART" ecalib -m1 ksp_full sens
"$BART" ones 3 128 128 1 m0 ; "$BART" resize -c 1 128 m0 mask   # simple mask stand-in (full)
CMD=(pics -S -i 100 -C 10 -R T:3:0:0.01 ksp_full sens rec)

runcol() {  # <wisdom 0|1>
  export BART_USE_FFTW_WISDOM=$1
  rm -rf "$BART_FFTW_WISDOM_DIR"; mkdir -p "$BART_FFTW_WISDOM_DIR"
  "$BART" "${CMD[@]}" >/dev/null 2>&1                       # run 1: discarded (seeds wisdom)
  local t
  for _ in 1 2 3 4 5; do
    t=$( { /usr/bin/time -f '%e' "$BART" "${CMD[@]}" >/dev/null; } 2>&1 | tail -1 )
    echo "$t"
  done | sort -n | awk -v w="$1" 'NR==1{b=$1} {s+=$1;n++} END{printf "WISDOM=%s  best=%.2fs  mean=%.2fs\n", w, b, s/n}'
}

echo "### bart_wisdom_probe  bart=$BART  OMP_NUM_THREADS=$OMP_NUM_THREADS"
runcol 0
runcol 1
