"""The long runs: chains of drawing networks. Results go to
files/chains.json, so the step files can read them in a second.

    python3 sweep.py            runs every chain that is not in the file
    python3 sweep.py plain      runs one chain by name

One generation: train a new network on the pile (about 6 seconds), let
it draw 1,797 pictures, and build the next pile.
"""
import json
import sys
from pathlib import Path

import numpy as np

from draw import different, draw, judge, names, off, real, train

FILE = Path(__file__).parent / 'files' / 'chains.json'
SHOW = 24        # pictures of every generation kept for looking at

# name: temperature, generations, share of real pictures in every pile,
#       pile up everything, seed
CHAINS = {
    'plain':        (1.0, 30, 0.0, False, 0),
    'careful':      (0.7, 20, 0.0, False, 0),
    'careful-b':    (0.7, 20, 0.0, False, 1),
    'careful-c':    (0.7, 20, 0.0, False, 2),
    'keep-10':      (0.7, 30, 0.1, False, 0),
    'keep-50':      (0.7, 30, 0.5, False, 0),
    'pile-up':      (0.7, 30, 0.0, True, 0),
    'plain-pile-up': (1.0, 30, 0.0, True, 0),
}


def run(name):
    """One chain. Returns a row for every generation."""
    temperature, generations, keep, pile_up, seed = CHAINS[name]
    originals, digits = real()
    n = len(originals)
    ref, _ = judge(originals, digits)
    dice = np.random.default_rng(100 + seed)
    pile, everything, rows = originals, originals, []
    for g in range(generations + 1):
        net = train(pile, 1000 * seed + g)
        drawn = draw(net, n, dice, temperature)
        seen = np.bincount(names(ref, drawn), minlength=10)
        rows.append({'generation': g, 'pile': len(pile),
                     'different': different(drawn),
                     'off': round(off(drawn, originals), 2),
                     'digits': [int(v) for v in seen],
                     'ink': round(float((drawn / 2).mean()), 4),
                     'show': [''.join(map(str, p)) for p in drawn[:SHOW]]})
        print(name, g, rows[-1]['different'], rows[-1]['off'],
              rows[-1]['digits'], flush=True)
        if pile_up:
            everything = np.concatenate([everything, drawn])
            pile = everything
        elif keep:
            k = int(n * keep)
            fresh = originals[dice.choice(n, k, replace=False)]
            pile = np.concatenate([drawn[:n - k], fresh])
        else:
            pile = drawn
    return rows


def done():
    return json.loads(FILE.read_text()) if FILE.exists() else {}


if __name__ == '__main__':
    todo = sys.argv[1:] or [c for c in CHAINS if c not in done()]
    for name in todo:
        rows = run(name)
        out = done()
        out[name] = rows
        FILE.write_text(json.dumps(out))
