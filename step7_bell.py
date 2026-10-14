"""Step 7. The model of the proof: a bell curve that learns from itself.

    python3 step7_bell.py
"""
import numpy as np

from bell import chain, spreads

SHOW = (0, 1, 10, 100, 500, 1000, 2000)

if __name__ == '__main__':
    print('one chain: 100 numbers in every pile')
    print('  model   middle   spread')
    models = chain(100, 2000, seed=0)
    for g in SHOW:
        print('  %5d   %6.2f   %.4f' % (g, *models[g]))
    print()
    print('1,000 chains: the spread of the model in the middle chain')
    print('  model      pile of 10   pile of 100   pile of 1,000')
    table = {n: spreads(n, 2000, 1000) for n in (10, 100, 1000)}
    for g in SHOW:
        print('  %5d      %10.4f   %11.4f   %13.4f' % (
            g, *(np.median(table[n][:, g]) for n in (10, 100, 1000))))
    print()
    for n in (10, 100, 1000):
        small = (table[n][:, -1] < 0.5).sum()
        print('  pile of %5d: %4d of 1,000 chains end below spread 0.5' % (
            n, small))
