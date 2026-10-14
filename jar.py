"""The smallest model there is: a jar of marbles.

Learning = counting how many marbles of each colour there are.
Making   = drawing marbles by those counts, with the marble put back.
The next model learns only from the marbles the last model made.
"""
import random
from collections import Counter

JAR = {'red': 10, 'blue': 5, 'green': 3, 'yellow': 1, 'purple': 1}


def marbles(counts):
    """The jar as a list: one entry for every marble."""
    return [colour for colour, n in counts.items() for _ in range(n)]


def learn(pile):
    """A model of a pile: the count of every colour that is in it."""
    return Counter(pile)


def make(model, n, dice):
    """Draw n marbles by the model's counts. A colour with count 0
    is not in the model, so it can never be drawn."""
    colours = list(model)
    return dice.choices(colours, weights=[model[c] for c in colours], k=n)


def chain(pile, dice, limit=100000, n=None):
    """Model after model, each one learning from the last one's marbles.
    Returns the pile of every generation, until one colour is left."""
    piles = [list(pile)]
    while len(set(piles[-1])) > 1 and len(piles) <= limit:
        piles.append(make(learn(piles[-1]), n or len(pile), dice))
    return piles


def missed(count, n):
    """Chance that a colour with `count` of n marbles is drawn 0 times
    in n draws."""
    return (1 - count / n) ** n


def many(chains=10000, seed=0, jar=None):
    """Run many chains. Who is the last colour, and after how long?"""
    dice = random.Random(seed)
    pile = marbles(jar or JAR)
    wins, lengths = Counter(), []
    for _ in range(chains):
        piles = chain(pile, dice)
        wins[piles[-1][0]] += 1
        lengths.append(len(piles) - 1)
    lengths.sort()
    return wins, lengths
