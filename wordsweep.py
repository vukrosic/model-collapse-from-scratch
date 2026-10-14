"""The long runs of the word model. Results go to files/words.json.

    python3 wordsweep.py         (about 10 minutes)

Every chain has 30 models. Model 1 reads the real text; model 2 reads
only what model 1 wrote, and so on. For every model: different words,
different pairs, and the first 30 words it wrote.
"""
import json
from collections import Counter
from pathlib import Path

from words import chain, kinds, pairs, real

FILE = Path(__file__).parent / 'files' / 'words.json'
MODELS = 30

# name: share of real text in every pile, pile up everything, times
CHAINS = {
    'replace': (0.0, False, 1),
    'keep-10': (0.1, False, 1),
    'pile-up': (0.0, True, 1),
    'times-10': (0.0, False, 10),
}
GROUPS = ((1, 1), (2, 9), (10, 99), (100, 10 ** 9))


def survivors(text, written):
    """For every group of words (by how often the real text uses
    them): how many of them the model still writes."""
    seen = Counter(text)
    used = set(written)
    return [sum(1 for w, n in seen.items() if lo <= n <= hi and w in used)
            for lo, hi in GROUPS]


def run(name, text):
    keep, pile_up, times = CHAINS[name]
    written = chain(text, MODELS - 1, keep=keep, pile_up=pile_up,
                    times=times)
    rows = []
    for g, out in enumerate(written):
        rows.append({'model': g + 1, 'different': kinds(out),
                     'pairs': pairs(out), 'groups': survivors(text, out),
                     'start': ' '.join(out[:30])})
        print(name, g + 1, rows[-1]['different'], flush=True)
    return rows


if __name__ == '__main__':
    text = real()
    out = json.loads(FILE.read_text()) if FILE.exists() else {}
    out['real'] = {'words': len(text), 'different': kinds(text),
                   'pairs': pairs(text), 'groups': [
                       sum(1 for n in Counter(text).values() if lo <= n <= hi)
                       for lo, hi in GROUPS]}
    for name in CHAINS:
        if name not in out:
            out[name] = run(name, text)
            FILE.write_text(json.dumps(out))
