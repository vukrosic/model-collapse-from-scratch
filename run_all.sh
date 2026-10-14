#!/bin/sh
# Runs every step, the checks and the exercise, in the order of the video.
# The long picture chains are read from files/chains.json and the word
# chains from files/words.json. To make them again: delete the file, then
#   python3 sweep.py        (about two hours on a laptop)
#   python3 wordsweep.py    (about ten minutes)
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in step1_jar.py step2_chains.py step3_words.py step4_wordchain.py \
         step5_draw.py step6_loop.py step7_bell.py step8_fix.py \
         step9_pile.py tests.py exercise.py; do
  echo
  echo "=== $f ==="
  $PY -W ignore $f
done
