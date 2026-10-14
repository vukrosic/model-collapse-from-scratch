"""Step 4. Model after model, each one reading only what the last wrote.

    python3 step4_wordchain.py        (about 20 seconds)
"""
from collections import Counter

from words import chain, kinds, pairs, real

MODELS = 30
SHOW = (1, 2, 3, 5, 10, 20, 30)


def top_share(text, n=100):
    """Share of the text that is made of its n most frequent words."""
    return sum(c for _, c in Counter(text).most_common(n)) / len(text)


if __name__ == '__main__':
    text = real()
    written = chain(text, MODELS - 1)   # written[0] is model 1
    print('             different   different   share of the 100')
    print('             words       pairs       most used words')
    print('  real text  %9s   %9s   %.2f' % (
        format(kinds(text), ','), format(pairs(text), ','),
        top_share(text)))
    for g in SHOW:
        out = written[g - 1]
        print('  model %2d   %9s   %9s   %.2f' % (
            g, format(kinds(out), ','), format(pairs(out), ','),
            top_share(out)))
    print()
    for g in (1, MODELS):
        print('model %d writes: %s' % (g, ' '.join(written[g - 1][:22])))
    first, last = set(text), set(written[-1])
    seen = Counter(text)
    print()
    print('which words of the real text does model %d still write?' %
          MODELS)
    for low, high, name in ((1, 1, 'used 1 time'), (2, 9, '2 to 9 times'),
                            (10, 99, '10 to 99 times'),
                            (100, 10 ** 9, '100 times or more')):
        group = [w for w, n in seen.items() if low <= n <= high]
        kept = sum(1 for w in group if w in last)
        print('  %-18s %6s words, still written: %6s' % (
            name, format(len(group), ','), format(kept, ',')))
