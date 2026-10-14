"""Step 6. The picture model in a loop: model 2 learns only from the
pictures of model 1, model 3 from model 2, and so on.

    python3 step6_loop.py

Reads files/chains.json. To run the chains again (about 2 hours on a
laptop): delete that file, then  python3 sweep.py
plain:   the dice pick by the network's chances (temperature 1)
careful: likely inks made more likely first (temperature 0.7)
"""
import json
from pathlib import Path

CHAINS = json.loads((Path(__file__).parent / 'files' / 'chains.json')
                    .read_text())
SHOW = (1, 2, 5, 10, 15, 20, 25, 30)


def picture(code):
    return [''.join(' .#'[int(v)] for v in code[r * 8:r * 8 + 8])
            for r in range(8)]


def side_by_side(codes):
    rows = [picture(c) for c in codes]
    return '\n'.join('   '.join(r[i] for r in rows) for i in range(8))


def table(name):
    rows = CHAINS[name]
    print('%s: %d models' % (name, len(rows)))
    print('  model   different pictures   squares from the closest   most'
          ' seen digit')
    print('          of 1,797             real picture               (judge)')
    for m in SHOW:
        if m > len(rows):
            continue
        r = rows[m - 1]
        top = max(range(10), key=lambda d: r['digits'][d])
        print('  %5d   %18s   %24.2f   %d: %s' % (
            m, format(r['different'], ','), r['off'], top,
            format(r['digits'][top], ',')))
    for m in (1, 10, len(rows)):
        print('\nmodel %d draws:' % m)
        print(side_by_side(rows[m - 1]['show'][:8]))
    print()


if __name__ == '__main__':
    for name in ('plain', 'careful'):
        table(name)
    print('careful, with other dice: different pictures at the last model')
    for name in ('careful', 'careful-b', 'careful-c'):
        if name in CHAINS:
            print('  %-10s %s' % (name, format(CHAINS[name][-1]['different'],
                                               ',')))
