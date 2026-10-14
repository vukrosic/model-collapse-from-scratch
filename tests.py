"""Checks of the claims of the video. python tests.py

Marbles, words and the bell curve run here. The picture chains are
read from files/chains.json and the long word chains from
files/words.json (python3 sweep.py, python3 wordsweep.py).
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import bell
import jar
import words
from draw import SQUARES, Net, different, off, real
from exercise import answer

FILES = Path(__file__).parent / 'files'
CHAINS = json.loads((FILES / 'chains.json').read_text())
WORDS = json.loads((FILES / 'words.json').read_text())
results = []


def check(text, ok):
    results.append(bool(ok))
    print(('ok    ' if ok else 'WRONG ') + text)


# the jar
pile = jar.marbles(jar.JAR)
check('the real jar: 20 marbles, 5 colours',
      len(pile) == 20 and len(set(pile)) == 5)
check('learning is counting', jar.learn(pile) == Counter(jar.JAR))
dice = random.Random(3)
first = jar.make(jar.learn(pile), 20, dice)
check('model 1 (dice 3): red 9, blue 7, green 3, yellow 0, purple 1',
      Counter(first) == Counter(red=9, blue=7, green=3, purple=1))
check('a colour with count 0 can never be drawn again',
      'yellow' not in jar.make(jar.learn(first), 100000, random.Random(0)))
check('chance to miss a 1-of-20 colour in 20 draws: (19/20)^20 = 0.358',
      abs(jar.missed(1, 20) - 0.358) < 0.001)
check('3 of 20: 0.039, 5 of 20: 0.003',
      round(jar.missed(3, 20), 3) == 0.039
      and round(jar.missed(5, 20), 3) == 0.003)
piles = jar.chain(pile, random.Random(3))
check('one chain (dice 3): one colour left at model 17',
      len(piles) - 1 == 17 and set(piles[-1]) == {'red'})
wins, lengths = jar.many()
check('10,000 chains: red last in 4,980, blue 2,554, yellow 455',
      (wins['red'], wins['blue'], wins['yellow']) == (4980, 2554, 455))
check('every chain ends with one colour', sum(wins.values()) == 10000)
check('half of the chains end within 25 models, the longest lasts 240',
      lengths[4999] <= 25 and lengths[-1] == 240)

# the words
text = words.real()
check('the text: 204,062 words, 12,631 different',
      (len(text), words.kinds(text)) == (204062, 12631))
nxt = Counter(b for a, b in zip(text, text[1:] + text[:1]) if a == 'first')
check('after "first": 362 times, 107 different words, citizen 43',
      (sum(nxt.values()), len(nxt), nxt['citizen']) == (362, 107, 43))
written = WORDS['replace']
check('model 1 writes 10,144 different words',
      written[0]['different'] == 10144)
check('model 1 loses 2,162 of the 5,704 words used once, none used 10+',
      WORDS['real']['groups'][0] - written[0]['groups'][0] == 2162
      and written[0]['groups'][2:] == WORDS['real']['groups'][2:])
check('model 30 writes 2,771 different words', written[29]['different']
      == 2771)
check('model 30 keeps all 270 words used 100 times or more, 328 of the'
      ' once-words', written[29]['groups'][3] == 270
      and written[29]['groups'][0] == 328)
check('different words never grow along the replace chain',
      all(a['different'] >= b['different']
          for a, b in zip(written, written[1:])))
check('10 % real text: model 30 writes 6,509 different words',
      WORDS['keep-10'][29]['different'] == 6509)
check('pile up: model 30 writes 9,447 different words',
      WORDS['pile-up'][29]['different'] == 9447)

# the pictures
pictures, digits = real()
check('1,797 real pictures of 64 squares, 3 inks',
      pictures.shape == (1797, SQUARES) and set(np.unique(pictures))
      == {0, 1, 2})
net = Net(2 * SQUARES, 3, 0)
check('the network has 99,587 numbers',
      sum(v.size for v in net.w + net.b) == 99587)
check('distance of the real pictures to themselves is 0',
      off(pictures[:50], pictures) == 0)
check('the real pictures are all different',
      different(pictures) == 1797)
check('model 1 (plain): 1,797 different pictures, 9.42 squares off',
      CHAINS['plain'][0]['different'] == 1797
      and CHAINS['plain'][0]['off'] == 9.42)

# the bell curve
s = bell.spreads(100, 1000, 1000)
check('pile of 100, 1,000 chains: median spread at model 1,000 below'
      ' 0.01', np.median(s[:, 1000]) < 0.01)
s = bell.spreads(1000, 2000, 1000)
check('pile of 1,000: median spread at model 2,000 is 0.37',
      round(float(np.median(s[:, 2000])), 2) == 0.37)

# the exercise
rows = answer()
check('exercise: 0 real marbles: 1,976 of 2,000 chains at one colour',
      rows[0][2] == 1976)
check('exercise: 1 real marble: 66 chains at one colour, 2.84 colours',
      rows[1][2] == 66 and round(rows[1][1], 2) == 2.84)
check('exercise: more real marbles keep more colours',
      all(a[1] < b[1] for a, b in zip(rows, rows[1:])))

# PICTURE-CHECKS
C = CHAINS
check('careful dice: model 21 draws 1 different picture',
      C['careful'][20]['different'] == 1)
check('10 % real pictures: model 10, 1,694 of 1,797 called a six',
      C['keep-10'][9]['digits'][6] == 1694)
check('half real pictures: model 30 draws 1,147 different pictures',
      C['keep-50'][29]['different'] == 1147)
check('pile up: the pile of model 30 is 30 times the real pictures',
      C['pile-up'][29]['pile'] == 30 * 1797)
check('pile up: model 30 draws 1,515 different pictures, 3 twos',
      C['pile-up'][29]['different'] == 1515
      and C['pile-up'][29]['digits'][2] == 3)
check('plain dice: model 30 is 13.16 squares off, 12.54 with piling up',
      C['plain'][29]['off'] == 13.16 and C['plain-pile-up'][29]['off'] == 12.54)
check('pile up: model 30 still writes 3,080 of the once-words',
      WORDS['pile-up'][29]['groups'][0] == 3080)

print('\n%d of %d checks ok' % (sum(results), len(results)))
assert all(results)
