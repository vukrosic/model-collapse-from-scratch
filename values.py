"""Every number the film shows, written to values.json.

    python3 values.py

Reads files/words.json (made by wordsweep.py) and files/chains.json (made
by sweep.py) and repeats the short measurements of the step files.
Takes about two minutes.
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from bell import chain as bell_chain, spreads
from draw import LEVELS, SQUARES, Net, judge, real as real_pictures
from exercise import FRESH, colours_left
from jar import JAR, chain, learn, many, marbles, missed
from step1_jar import ORDER, SEED
from step4_wordchain import top_share
from step5_draw import apart, careful
from words import after, chain as word_chain, learn as learn_words, real

FILES = Path(__file__).parent / 'files'
WORDS = json.loads((FILES / 'words.json').read_text())
CHAINS = json.loads((FILES / 'chains.json').read_text())


def r3(v):
    """Three digits that count, as a plain float."""
    return float('%.3g' % v)


def jar_part():
    piles = chain(marbles(JAR), random.Random(SEED))
    counts = [[learn(p)[c] for c in ORDER] for p in piles]
    wins, lengths = many(10000)
    n = len(lengths)
    return {'order': ORDER, 'jar': [JAR[c] for c in ORDER],
            'draws': piles[1], 'counts': counts,
            'missed': {c: r3(missed(JAR[c], 20)) for c in ORDER},
            'wins': [wins[c] for c in ORDER],
            'lengths': {'half': lengths[n // 2], 'fast': lengths[n // 10],
                        'slow': lengths[9 * n // 10], 'longest': lengths[-1]}}


def words_part():
    text = real()
    table = learn_words(text)
    follow = {w: after(table, w)[:6] for w in ('first', 'citizen')}
    total = {w: sum(n for _, n in after(table, w)) for w in follow}
    written = word_chain(text, 29)
    seen = Counter(text)
    rare = {w for w, n in seen.items() if n == 1}
    used = set(written[0])
    return {'real': WORDS['real'], 'follow': follow, 'total': total,
            'next': {w: len(after(table, w)) for w in follow},
            'lost_once': len(rare - used),
            'lost_common': sum(1 for w, n in seen.items()
                               if n >= 10 and w not in used),
            'common': sum(1 for n in seen.values() if n >= 10),
            'share': {'real': r3(top_share(text)),
                      'model1': r3(top_share(written[0])),
                      'model30': r3(top_share(written[-1]))},
            'chains': {k: [{key: r[key] for key in
                            ('model', 'different', 'pairs', 'groups')}
                           for r in v] for k, v in WORDS.items()
                       if k != 'real'},
            'lines': {'model1': ' '.join(written[0][:14]),
                      'model30': ' '.join(written[-1][:14])}}


def pictures_part():
    pictures, digits = real_pictures()
    net = Net(2 * SQUARES, LEVELS, 0)
    _, right = judge(pictures, digits)
    chances = np.array([0.7, 0.2, 0.1])
    out = {'real': {'count': len(pictures),
                    'digits': np.bincount(digits).tolist(),
                    'show': [''.join(str(int(v)) for v in p)
                             for p in pictures[:24]],
                    'apart': r3(apart(pictures))},
           'network': int(sum(v.size for v in net.w + net.b)),
           'judge': right,
           'careful': [r3(v) for v in careful(chances, 0.7)],
           'chains': {}}
    for name, rows in CHAINS.items():
        out['chains'][name] = [{'model': r['generation'] + 1,
                                'pile': r['pile'],
                                'different': r['different'],
                                'off': r['off'], 'digits': r['digits'],
                                'show': r['show']} for r in rows]
    # one picture of model 1 and the real picture closest to it
    drawn = np.array([int(c) for c in CHAINS['plain'][0]['show'][0]])
    gaps = (pictures != drawn).sum(axis=1)
    near = int(gaps.argmin())
    out['example'] = {'drawn': CHAINS['plain'][0]['show'][0],
                      'real': ''.join(str(int(v)) for v in pictures[near]),
                      'distance': int(gaps[near])}
    return out


def first_piles():
    """The 100 numbers model 1 learns from, and the 100 it makes:
    the same dice as the first step of bell_chain(100, 2000, seed=0)."""
    dice = np.random.default_rng(0)
    p0 = dice.normal(0.0, 1.0, 100)
    middle, spread = p0.mean(), p0.std(ddof=1)
    p1 = dice.normal(middle, spread, 100)
    return [round(float(v), 2) for v in p0], [round(float(v), 2) for v in p1]


def bell_part():
    models = bell_chain(100, 2000, seed=0)
    show = (0, 1, 10, 100, 500, 1000, 2000)
    table = {n: spreads(n, 2000, 1000) for n in (10, 100, 1000)}
    return {'chain': [[g, r3(models[g][1])] for g in show],
            'curve': [r3(models[g][1]) for g in range(0, 2001, 10)],
            'first': [r3(models[g][1]) for g in range(0, 21)],
            'pile0': first_piles()[0], 'pile1': first_piles()[1],
            'mid1': r3(models[1][0]),
            'piles': {str(n): [r3(np.median(table[n][:, g])) for g in show]
                      for n in table}, 'show': show,
            'pcurve': {str(n): [r3(np.median(table[n][:, g]))
                                for g in range(0, 2001, 25)] for n in table}}


def fix_part():
    return {'marbles': {str(f): colours_left(f) for f in (0, 1)},
            'exercise': {str(f): colours_left(f) for f in FRESH}}


if __name__ == '__main__':
    values = {'jar': jar_part(), 'words': words_part(),
              'pictures': pictures_part(), 'bell': bell_part(),
              'fix': fix_part()}
    Path(__file__).with_name('values.json').write_text(
        json.dumps(values, indent=1))
    print('values.json written,', len(json.dumps(values)), 'characters')
