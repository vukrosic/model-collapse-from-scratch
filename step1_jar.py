"""Step 1. A jar of 20 marbles, and three models, one after the other.

    python3 step1_jar.py
"""
import random

from jar import JAR, learn, make, marbles, missed

ORDER = list(JAR)
SEED = 3


def line(pile):
    counts = learn(pile)
    return '  '.join('%s %2d' % (c, counts[c]) for c in ORDER)


if __name__ == '__main__':
    dice = random.Random(SEED)
    pile = marbles(JAR)
    print('the real jar: %d marbles, %d colours' % (len(pile), len(JAR)))
    print('  real     ', line(pile))
    for g in range(1, 4):
        model = learn(pile)                 # count the colours
        pile = make(model, 20, dice)        # draw 20 marbles by the counts
        print('  model %d   ' % g, line(pile))
    print()
    print('a colour with count 0 is not in the next model: it cannot')
    print('come back.')
    print()
    print('chance that a colour is drawn 0 times in 20 draws')
    for count in (1, 3, 5, 10):
        print('  %2d of 20 marbles: %.3f' % (count, missed(count, 20)))
