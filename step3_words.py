"""Step 3. A model that writes text: a table of word pairs.

    python3 step3_words.py
"""
import random
from collections import Counter

from words import START, after, kinds, learn, real, write

SEED = 1


def once(text):
    """How many different words occur exactly one time."""
    return sum(1 for n in Counter(text).values() if n == 1)


if __name__ == '__main__':
    text = real()
    print('the real text: %s words, %s different' % (
        format(len(text), ','), format(kinds(text), ',')))
    print('  words used only one time: %s' % format(once(text), ','))
    print('  the start: ' + ' '.join(text[:12]))
    print()
    table = learn(text)
    for word in ('first', 'citizen'):
        rows = after(table, word)
        total = sum(n for _, n in rows)
        print('after "%s" (%d times in the text), %d different words:' % (
            word, total, len(rows)))
        print('  ' + ',  '.join('%s %d' % p for p in rows[:6]))
    print()
    dice = random.Random(SEED)
    out = write(table, len(text), dice)
    print('model 1 writes %s words, starting after "%s":' % (
        format(len(out), ','), START))
    print('  ' + ' '.join(out[:24]))
    print('  different words: %s (the real text: %s)' % (
        format(kinds(out), ','), format(kinds(text), ',')))
    seen = Counter(text)
    rare = [w for w, n in seen.items() if n == 1]
    used = set(out)
    back = sum(1 for w in rare if w in used)
    print('  of the %s words used one time, model 1 wrote %s' % (
        format(len(rare), ','), format(back, ',')))
    print('  and lost %s' % format(len(rare) - back, ','))
    common = [w for w, n in seen.items() if n >= 10]
    print('  of the %s words used 10 times or more, it lost %d' % (
        format(len(common), ','), sum(1 for w in common if w not in used)))
