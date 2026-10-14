"""Step 8. The first fix: put real data into every pile.

    python3 step8_fix.py        (about 30 seconds)

Marbles: run here. Words and pictures: read from files/words.json and
files/chains.json (python3 wordsweep.py, python3 sweep.py).
"""
import json
from pathlib import Path

from exercise import colours_left

FILES = Path(__file__).parent / 'files'
WORDS = json.loads((FILES / 'words.json').read_text())
CHAINS = json.loads((FILES / 'chains.json').read_text())

if __name__ == '__main__':
    print('marbles: 2,000 chains of 100 models, 20 marbles in every pile')
    for fresh in (0, 1):
        left, one = colours_left(fresh)
        print('  %d real marble(s) in every pile: %.2f colours left on'
              ' average, %s chains down to one colour' % (
                  fresh, left, format(one, ',')))
    print()
    print('words: different words, 30 models (real text: %s)' %
          format(WORDS['real']['different'], ','))
    print('  model   replace   10 %% real')
    for m in (1, 2, 5, 10, 20, 30):
        print('  %5d   %7s   %9s' % (
            m, format(WORDS['replace'][m - 1]['different'], ','),
            format(WORDS['keep-10'][m - 1]['different'], ',')))
    print('  words used once, still written by model 30: %s (replace),'
          ' %s (10 %% real)' % (WORDS['replace'][-1]['groups'][0],
                                WORDS['keep-10'][-1]['groups'][0]))
    print()
    print('pictures (careful dice): different pictures of 1,797')
    names = ('careful', 'keep-10', 'keep-50')
    print('  model   ' + '   '.join('%9s' % n for n in names))
    for m in (1, 5, 10, 15, 20, 25, 30):
        cells = []
        for n in names:
            rows = CHAINS[n]
            cells.append(format(rows[m - 1]['different'], ',')
                         if m <= len(rows) else '')
        print('  %5d   ' % m + '   '.join('%9s' % c for c in cells))
