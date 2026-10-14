"""Exercise. Put real marbles back into every pile of the jar chain.

    python3 exercise.py

In step 2 every model learned only from the 20 marbles the last model
made, and every chain ended with one colour. Here a pile is made of
`fresh` marbles taken from the real jar and 20 - fresh marbles made by
the last model. Your part: the marked lines in `next_pile`.
"""
import random

from jar import JAR, learn, make, marbles

MODELS = 100
CHAINS = 2000
FRESH = (0, 1, 2, 5, 10)


def next_pile(pile, real, fresh, dice):
    # your part starts here
    made = make(learn(pile), len(pile) - fresh, dice)
    return made + dice.sample(real, fresh)
    # and ends here


def colours_left(fresh, seed=0):
    """Average number of colours in the pile of model 100, and in how
    many of the chains all 5 colours were lost down to one."""
    dice = random.Random(seed)
    real = marbles(JAR)
    total, one = 0, 0
    for _ in range(CHAINS):
        pile = real
        for _ in range(MODELS):
            pile = next_pile(pile, real, fresh, dice)
        total += len(set(pile))
        one += len(set(pile)) == 1
    return total / CHAINS, one


def answer():
    return [(fresh, *colours_left(fresh)) for fresh in FRESH]


if __name__ == '__main__':
    print('20 marbles in every pile, %d models, %s chains' % (
        MODELS, format(CHAINS, ',')))
    print('  real marbles   colours in the pile   chains with')
    print('  in every pile  of model %d (average)  one colour' % MODELS)
    for fresh, left, one in answer():
        print('  %2d of 20        %.2f                  %5s' % (
            fresh, left, format(one, ',')))
    print()
    print('the real jar itself: a pile of 20 has 5 colours.')
