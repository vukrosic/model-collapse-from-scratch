"""A model that writes: a table of word pairs.

Learning = for every word, count which words came right after it.
Writing  = start at a word, pick the next word by those counts, repeat.
The text is files/shakespeare.txt (about 1 MB of plays, public domain).
"""
import bisect
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

TEXT = Path(__file__).parent / 'files' / 'shakespeare.txt'
START = 'first'          # the first word of the real text


def real():
    """The real text as a list of words, lower case, no punctuation."""
    return re.findall(r"[a-z']+", TEXT.read_text().lower())


def learn(text):
    """The pair table. table[word] = (next words, running sum of counts).
    The text is treated as a ring, so the last word has a next word."""
    counts = defaultdict(Counter)
    for word, nxt in zip(text, text[1:] + text[:1]):
        counts[word][nxt] += 1
    table = {}
    for word, after in counts.items():
        total, sums = 0, []
        for n in after.values():
            total += n
            sums.append(total)
        table[word] = (list(after), sums)
    return table


def after(table, word):
    """What followed `word`, most frequent first: [(next word, count)]."""
    nxt, sums = table[word]
    counts = [b - a for a, b in zip([0] + sums[:-1], sums)]
    return sorted(zip(nxt, counts), key=lambda p: -p[1])


def write(table, n, dice, start=START):
    """Write n words. Each next word is picked by the counts in the table:
    a pair seen 7 times is 7 times as likely as a pair seen once."""
    word, out = start, []
    for _ in range(n):
        nxt, sums = table[word]
        word = nxt[bisect.bisect_right(sums, dice.random() * sums[-1])]
        out.append(word)
    return out


def kinds(text):
    """How many different words a text has."""
    return len(set(text))


def pairs(text):
    """How many different word pairs a text has."""
    return len(set(zip(text, text[1:])))


def chain(text, generations, seed=1, keep=0.0, pile_up=False, times=1):
    """Model after model. Returns what every model wrote (same length as
    the real text), starting with model 1, which read the real text.

    keep:    share of every new pile that is real text again (0 to 1)
    pile_up: every pile is the real text plus everything written so far
    times:   each model writes this many times as much for the next one
    """
    dice = random.Random(seed)
    n = len(text)
    pile, everything, written = list(text), list(text), []
    for _ in range(generations + 1):
        out = write(learn(pile), n * times, dice)
        written.append(out[:n])
        if pile_up:
            everything = everything + out
            pile = everything
        elif keep:
            k = int(n * keep)
            at = dice.randrange(n - k)
            pile = out[:n * times - k] + text[at:at + k]
        else:
            pile = out
    return written
