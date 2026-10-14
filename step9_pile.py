"""Step 9. The second fix: pile up, never throw data away.

    python3 step9_pile.py

Every pile is the real data plus everything every model before made.
Reads files/words.json and files/chains.json.
"""
import json
from pathlib import Path

FILES = Path(__file__).parent / 'files'
WORDS = json.loads((FILES / 'words.json').read_text())
CHAINS = json.loads((FILES / 'chains.json').read_text())

if __name__ == '__main__':
    print('words: different words (real text: %s)' %
          format(WORDS['real']['different'], ','))
    print('  model   replace   pile up   write 10 times as much')
    for m in (1, 2, 5, 10, 20, 30):
        print('  %5d   %7s   %7s   %9s' % tuple(
            [m] + [format(WORDS[n][m - 1]['different'], ',')
                   for n in ('replace', 'pile-up', 'times-10')]))
    print('  words used once, still written by model 30: %s (pile up)' %
          WORDS['pile-up'][-1]['groups'][0])
    print()
    print('pictures: different pictures of 1,797, and squares from the'
          ' closest real picture')
    names = ('plain', 'plain-pile-up', 'careful', 'pile-up')
    print('  model   ' + '   '.join('%16s' % n for n in names))
    for m in (1, 5, 10, 20, 30):
        cells = []
        for n in names:
            rows = CHAINS[n]
            cells.append('%s / %.1f' % (format(rows[m - 1]['different'], ','),
                                        rows[m - 1]['off'])
                         if m <= len(rows) else '')
        print('  %5d   ' % m + '   '.join('%16s' % c for c in cells))
