"""Step 5. A picture model: a network that draws digits square by square.

    python3 step5_draw.py        (about 20 seconds)

Model 1 learns from the 1,797 real pictures and draws as many new ones.
Two scores for the drawn pictures: which digit the judge sees, and how
many squares differ from the closest real picture.
"""
import numpy as np

from draw import (LEVELS, SQUARES, Net, different, draw, judge, names, off,
                  real, train)


def show(picture):
    return '\n'.join('    ' + ''.join(' .#'[v] for v in picture[r * 8:r * 8 + 8])
                     for r in range(8))


def apart(pictures):
    """For every real picture: squares that differ from the closest
    OTHER real picture. Returns the average."""
    out = []
    for start in range(0, len(pictures), 200):
        d = (pictures[start:start + 200, None, :] != pictures[None]).sum(2)
        d[np.arange(len(d)), np.arange(start, start + len(d))] = SQUARES + 1
        out.append(d.min(1))
    return float(np.concatenate(out).mean())


def careful(chances, temperature):
    """The same as Net.chances does: raise to 1 / temperature, share again."""
    c = np.array(chances) ** (1 / temperature)
    return c / c.sum()


if __name__ == '__main__':
    pictures, digits = real()
    print('real pictures: %s, inks: white is a space, grey ., black #' %
          format(len(pictures), ','))
    print('the first real picture, digit %d:' % digits[0])
    print(show(pictures[0]))
    net = Net(2 * SQUARES, LEVELS, 0)
    print('numbers in the network: %s' % format(
        sum(v.size for v in net.w + net.b),
        ','))
    print()
    print('careful dice: chances 70 / 20 / 10 % at temperature 0.7 become',
          ' / '.join('%.0f' % (100 * v) for v in careful((.7, .2, .1), .7)),
          '%')
    print()
    ref, right = judge(pictures, digits)
    print('the judge names %d of 500 real pictures it never saw' % right)
    print('a real picture differs from the closest other real picture in')
    print('  %.2f squares (average)' % apart(pictures))
    net = train(pictures, 0)
    drawn = draw(net, len(pictures), np.random.default_rng(100))
    print('model 1 drew %s pictures, %s of them different' % (
        format(len(drawn), ','), format(different(drawn), ',')))
    print('  squares that differ from the closest real picture: %.2f' %
          off(drawn, pictures))
    print('  digits the judge sees:', np.bincount(names(ref, drawn),
                                                 minlength=10).tolist())
    print('the first three pictures of model 1:')
    for p in drawn[:3]:
        print(show(p))
        print()
