"""The model of the proof: a bell curve.

Learning = measure the middle and the spread of the numbers in a pile.
Making   = throw dice for new numbers from the bell curve with that
           middle and that spread.
"""
import numpy as np


def learn(pile):
    """The middle (the average) and the spread (how far a typical
    number is from the middle)."""
    return float(pile.mean()), float(pile.std(ddof=1))


def make(model, n, dice):
    middle, spread = model
    return dice.normal(middle, spread, n)


def chain(n, generations, seed):
    """Start: middle 0, spread 1. Returns the model of every generation."""
    dice = np.random.default_rng(seed)
    model, models = (0.0, 1.0), [(0.0, 1.0)]
    for _ in range(generations):
        model = learn(make(model, n, dice))
        models.append(model)
    return models


def spreads(n, generations, chains, seed=0):
    """Many chains at once. Returns the spread of every chain at every
    generation: an array of chains by generations + 1."""
    dice = np.random.default_rng(seed)
    middle, spread = np.zeros(chains), np.ones(chains)
    out = [spread.copy()]
    for _ in range(generations):
        pile = dice.normal(middle[:, None], spread[:, None], (chains, n))
        middle, spread = pile.mean(1), pile.std(1, ddof=1)
        out.append(spread.copy())
    return np.array(out).T
