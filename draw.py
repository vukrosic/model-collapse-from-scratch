"""A network that draws 8 by 8 pictures of digits, one square at a time.

A picture is 64 squares. Every square is white (0), grey (1) or black (2).
The network is asked one question, 64 times for every picture:
"these squares are drawn already. What is the next square?"
It answers with three chances: white, grey, black. Dice pick by them.
A language model does the same thing with words in place of squares.
Numpy only.
"""
import numpy as np
from sklearn.datasets import load_digits

SQUARES = 64
LEVELS = 3        # white, grey, black
HIDDEN = 256
STEPS = 8000      # learning steps in one training
BATCH = 256       # questions in one learning step
RATE = 0.001


def real():
    """The 1,797 real pictures and the digit each one shows.
    On disk a square has 17 greys (0 to 16). Here: 0-4 white, 5-11 grey,
    12-16 black."""
    data = load_digits()
    grey = data.data
    return ((grey >= 5).astype(int) + (grey >= 12).astype(int)), data.target


class Net:
    """Three tables of weights. Negative sums are set to zero between
    them. The last table gives one score for every possible answer."""

    def __init__(self, n_in, n_out, seed, hidden=HIDDEN):
        dice = np.random.default_rng(seed)
        sizes = [n_in, hidden, hidden, n_out]
        self.w = [dice.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.w[-1] /= np.sqrt(2)
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.w + self.b]
        self.v = [np.zeros_like(p) for p in self.w + self.b]
        self.t = 0

    def scores(self, x):
        """Push the inputs through. Returns what every table gave."""
        a1 = np.maximum(0, x @ self.w[0] + self.b[0])
        a2 = np.maximum(0, a1 @ self.w[1] + self.b[1])
        return a1, a2, a2 @ self.w[2] + self.b[2]

    def chances(self, x, temperature=1.0):
        """Scores turned into chances that add up to 1.
        temperature 1: as the scores say. Below 1: the likely answers
        get more likely, the rare ones rarer."""
        s = self.scores(x)[2] / temperature
        s = s - s.max(1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(1, keepdims=True)

    def step(self, x, answer):
        """One learning step (Adam): move every weight a little so that
        the right answers get a higher chance. Returns the loss."""
        a1, a2, _ = self.scores(x)
        p = self.chances(x)
        rows = np.arange(len(answer))
        loss = float(-np.log(p[rows, answer] + 1e-12).mean())
        d3 = p
        d3[rows, answer] -= 1
        d3 = d3 / np.float32(len(answer))
        d2 = (d3 @ self.w[2].T) * (a2 > 0)
        d1 = (d2 @ self.w[1].T) * (a1 > 0)
        moves = [x.T @ d1, a1.T @ d2, a2.T @ d3,
                 d1.sum(0), d2.sum(0), d3.sum(0)]
        self.t += 1
        for i, (p_, g) in enumerate(zip(self.w + self.b, moves)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            m = self.m[i] / (1 - 0.9 ** self.t)
            v = self.v[i] / (1 - 0.999 ** self.t)
            p_ -= (RATE * m / (np.sqrt(v) + 1e-8)).astype(np.float32)
        return loss


def question(pictures, k):
    """What the network sees when square k is next: the squares before
    k (the others hidden), and 64 marks that say which are drawn."""
    drawn = np.arange(SQUARES)[None, :] < np.reshape(k, (-1, 1))
    drawn = np.broadcast_to(drawn, pictures.shape)
    return np.concatenate([pictures / 2 * drawn, drawn], 1).astype(np.float32)


def train(pictures, seed, steps=STEPS):
    """A new network that learns to draw from `pictures` only."""
    dice = np.random.default_rng(seed)
    net = Net(2 * SQUARES, LEVELS, seed)
    with np.errstate(all='ignore'):
        for _ in range(steps):
            pick = pictures[dice.integers(0, len(pictures), BATCH)]
            k = dice.integers(0, SQUARES, BATCH)
            net.step(question(pick, k), pick[np.arange(BATCH), k])
    return net


def draw(net, n, dice, temperature=1.0):
    """n new pictures, square by square, left to right, top to bottom."""
    pictures = np.zeros((n, SQUARES), int)
    with np.errstate(all='ignore'):
        for k in range(SQUARES):
            p = net.chances(question(pictures, k), temperature)
            throw = dice.random((n, 1))
            pictures[:, k] = (throw > np.cumsum(p, 1)).sum(1).clip(0, 2)
    return pictures


def half(pictures):
    """White 0, grey 0.5, black 1."""
    return (pictures / 2).astype(np.float32)


def judge(pictures, digits, seed=0):
    """A second network, trained on real pictures and their digits. It
    names the digit in a picture. Returns (net, right of 500 unseen)."""
    dice = np.random.default_rng(seed)
    order = dice.permutation(len(digits))
    learn, test = order[:-500], order[-500:]
    net = Net(SQUARES, 10, seed, hidden=128)
    with np.errstate(all='ignore'):
        for _ in range(2000):
            pick = dice.choice(learn, 64)
            net.step(half(pictures[pick]), digits[pick])
        named = net.scores(half(pictures[test]))[2].argmax(1)
    return net, int((named == digits[test]).sum())


def names(net, pictures):
    """The digit the judge sees in every picture."""
    with np.errstate(all='ignore'):
        return net.scores(half(pictures))[2].argmax(1)


def different(pictures):
    """How many different pictures there are in a pile."""
    return len(np.unique(pictures, axis=0))


def off(pictures, originals):
    """For every picture: in how many squares it differs from the
    closest real picture. Returns the average."""
    out = []
    for start in range(0, len(pictures), 200):
        part = pictures[start:start + 200]
        out.append((part[:, None, :] != originals[None, :, :])
                   .sum(2).min(1))
    return float(np.concatenate(out).mean())
