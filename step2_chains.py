"""Step 2. The chain goes on until one colour is left. Then 10,000 chains.

    python3 step2_chains.py
"""
import random

from jar import JAR, chain, learn, many, marbles
from step1_jar import ORDER, SEED, line


def left(pile):
    return len(set(pile))


if __name__ == '__main__':
    piles = chain(marbles(JAR), random.Random(SEED))
    print('one chain (the dice of step 1)')
    for g, pile in enumerate(piles):
        print('  model %2d   %s   colours left: %d' % (g, line(pile),
                                                      left(pile)))
    print()
    wins, lengths = many(10000)
    n = len(lengths)
    print('10,000 chains from the same jar')
    print('  the last colour      chains   share at the start')
    for colour in ORDER:
        print('  %-8s            %6d   %2d of 20 = %2d %%' % (
            colour, wins[colour], JAR[colour], JAR[colour] * 5))
    print()
    print('  models until one colour is left')
    print('    half of the chains need at most %d' % lengths[n // 2])
    print('    the fastest tenth: at most %d' % lengths[n // 10])
    print('    the slowest tenth: at least %d' % lengths[9 * n // 10])
    print('    the longest chain: %d' % lengths[-1])
    print('  no chain kept two colours.')
