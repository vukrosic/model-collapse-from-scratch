# Model collapse from scratch

Code for the video **What Happens When AI Learns From AI**.
A model learns from real data and makes new data. The next model learns only from what the last one made, and so on. This code runs that loop on three models small enough to follow by hand: a jar of marbles, a table of word pairs that writes Shakespeare-like text, and a small network that draws 8 by 8 digits. The rare things go first: the yellow marble, the words Shakespeare used once. Then the rest narrows to copies of one picture or turns into noise. Two changes slow it down: put some real data back into every pile, or never throw old data away.
Nine small programs build this from an empty file. About 1,200 lines of Python 3. The name and the first measurements are from Shumailov, Shumaylov, Zhao, Papernot, Anderson and Gal, "AI models collapse when trained on recursively generated data", Nature, 2024.

```bash
pip install numpy scikit-learn
sh run_all.sh         # nine steps, the tests and the exercise; expected output is in demo-output.txt
```

`run_all.sh` takes about one minute on a laptop. The long chains are saved, so the steps only read them: the picture chains in `files/chains.json` (8 chains, up to 31 networks each) and the word chains in `files/words.json`. To make them again, delete the file and run `python3 sweep.py` (about two hours, no GPU) or `python3 wordsweep.py` (about ten minutes).

There is no data to download. The text is `files/shakespeare.txt`: the 1 MB "tiny Shakespeare" file of Andrej Karpathy's char-rnn, lines from the plays, public domain. The pictures are the digits set that comes with scikit-learn: 1,797 handwritten digits of 8 by 8 squares. The network is numpy only.

```python
def make(model, n, dice):
    """Draw n marbles by the model's counts. A colour with count 0
    is not in the model, so it can never be drawn."""
    colours = list(model)
    return dice.choices(colours, weights=[model[c] for c in colours], k=n)

pile = make(learn(pile), 20, dice)      # one model: count the pile, draw a new pile of 20
```

## Words

| In the video | In the paper | In the code |
| --- | --- | --- |
| model | generative model | `jar.learn` + `jar.make`, `words.learn` + `words.write`, `draw.train` + `draw.draw` |
| model number n | generation n | `g`, `generation` |
| what a model made | synthetic data | `drawn`, `out` |
| the real jar, the real text, the real pictures | original data | `JAR`, `words.real()`, `draw.real()` |
| the rare ones | tails of the distribution | colours with one marble; words used one time |
| one colour left, for ever | absorbing state | `len(set(pile)) == 1` |
| the spread | variance (its square root) | `bell.learn` |
| careful dice | sampling temperature below 1 | `temperature=0.7` |
| replace | each pile is only what the last model made | `keep=0, pile_up=False` |
| keep real data | mixing in original data | `keep=0.1` |
| pile up | accumulate | `pile_up=True` |
| different pictures | diversity | `draw.different` |
| squares from the closest real picture | distance to the data | `draw.off` |
| the judge | a classifier | `draw.judge`: a network that names the digit of a picture |

## Files

| File | Idea | What the run shows |
| --- | --- | --- |
| `jar.py` | The smallest model: a jar | Learning is counting the colours, making is drawing by the counts. 55 lines. |
| `step1_jar.py` | Three models, one after the other | Real jar: red 10, blue 5, green 3, yellow 1, purple 1. Model 1 draws no yellow, so no later model can. A colour with 1 of 20 marbles is missed by 20 draws with a chance of 0.358. |
| `step2_chains.py` | Until one colour is left | One chain ends with only red at model 17. 10,000 chains: every one ends with one colour. Red is last in 4,980 (it was 50 % of the jar), yellow in 455 (5 %). Half of the chains end within 25 models; the longest lasts 240. |
| `words.py` | A model that writes | A table of word pairs: for every word, which words came next and how often. |
| `step3_words.py` | One model writes | The text: 204,062 words, 12,631 different, 5,704 used only once. After "first": citizen 43, senator 33, murderer 33. Model 1 writes 10,144 different words: it loses 2,162 of the once-words and none of the 1,893 words used 10 times or more. |
| `step4_wordchain.py` | Model after model | Different words: 12,631 real, 8,756 at model 2, 4,898 at model 10, 2,771 at model 30. Model 30 still writes all 270 words used 100 times or more, and 328 of the 5,704 once-words. |
| `draw.py` | A network that draws | 64 questions per picture: "these squares are drawn, what is the next one?" White, grey or black. 99,587 numbers. Plus the judge, and the distance to the closest real picture. |
| `step5_draw.py` | Model 1 draws | 1,797 pictures, all different. 9.42 squares differ from the closest real picture; real pictures differ from each other in 7.35. |
| `step6_loop.py` | The pictures in a loop | Plain dice: the pictures turn to noise, 13.16 squares off at model 30, still all different. Careful dice: 515 different at model 5, 13 at model 10, 1 from model 15 on: copies of one digit. |
| `bell.py`, `step7_bell.py` | The model of the proof: a bell curve | Piles of 100 numbers, 1,000 chains: the middle chain has spread 0.58 at model 100 and 0.006 at model 1,000. Piles of 1,000 shrink slower: 0.37 at model 2,000. |
| `step8_fix.py` | Fix 1: real data in every pile | Marbles: 1 real marble of 20 leaves 2.84 colours at model 100 against 1.01. Words: 10 % real text keeps 6,509 different words at model 30 against 2,771. Pictures with careful dice: 10 % real is not enough (1,694 of 1,797 called a six at model 10); half real keeps 1,147 different at model 30. |
| `step9_pile.py` | Fix 2: never throw data away | Every pile is the real data plus all that was made so far; model 30 learns from 30 times the real data. Words: 9,447 different at model 30, and 3,080 of the once-words. Pictures with careful dice: 1,515 different at model 30, 7.1 squares off. |
| `sweep.py`, `wordsweep.py` | The saved chains | Run what is missing in `files/chains.json` and `files/words.json`. |
| `values.py` | Numbers for the pictures of the video | Writes `values.json` for the film project. Not part of `run_all.sh`. |
| `tests.py` | Checks of the claims | 36 checks. |
| `exercise.py` | Real marbles in every pile | See below. |

## Reading the numbers

- **Small models, one small set of data.** 20 marbles, 204,062 words, 1,797 pictures. Every model learns from a pile of the same size as the real data, unless the line says "pile up" or "10 times as much".
- **Dice.** Every chain has its own seed, so a run gives the same numbers again. `step6_loop.py` also runs the careful chain with two other seeds: all three end at 1 different picture.
- **Two ways to collapse.** With plain dice (temperature 1) the network keeps drawing different pictures, but they move away from real digits: from 9.42 to 13.16 squares off. With careful dice (temperature 0.7: chances of 70 / 20 / 10 % become 81 / 14 / 5 %) the pictures stay close to real digits but turn into copies. The paper's language models use beam search, which is careful too.
- **The judge.** A separate network trained on 1,297 real pictures; it names 485 of 500 others right. "Called a six" means the judge says so.
- **Writing more is not piling up.** Model after model writing 10 times as much for the next one keeps 7,461 different words at model 30; piling up keeps 9,447. Only the pile keeps the real text.
- **The bell.** The spread of one chain is noisy. The numbers in the table are the middle of 1,000 chains: half of the chains are above, half below.

## What this is not

- **Not a test of today's large models.** The papers, and this code, test small models in a closed loop where only what models made goes back in. Nothing here says that large language models have collapsed.
- **Not a measure of the web.** The share of text on the web written by models is not measured here.
- **Not a safe share of real data.** 10 % was enough for the marbles and the text, not for the pictures with careful dice. The share needed depends on the model and the data.
- **Not a reproduction of the paper.** The Nature paper fine-tunes OPT-125m on WikiText-2 and measures perplexity; with no original data it gets worse by 20 to 28 points, with 10 % kept only a little.

Sources: Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal, "AI models collapse when trained on recursively generated data", Nature 631, 755–759, 2024, and its earlier version "The Curse of Recursion: Training on Generated Data Makes Models Forget", arXiv 2305.17493, 2023. Alemohammad and others, "Self-Consuming Generative Models Go MAD", arXiv 2307.01850, 2023. Gerstgrasser and others, "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data", arXiv 2404.01413, 2024. The pictures: `sklearn.datasets.load_digits`, from the UCI set "Optical Recognition of Handwritten Digits".

## Try it

In step 2 every pile was 20 marbles made by the last model, and every chain ended with one colour. In `exercise.py` a pile is `fresh` marbles taken from the real jar and `20 - fresh` marbles made by the model. The function to write:

```python
def next_pile(pile, real, fresh, dice):
    # your part starts here
    # fresh marbles from the real jar,
    # 20 - fresh made by the model
    # and ends here
```

The file has an answer between the two marked lines; delete it and write your own first. Over 2,000 chains of 100 models, 0 real marbles leave 1.01 colours on average and 1 real marble leaves 2.84. Guess first: how many colours are left with 2, 5 and 10 real marbles of 20? Then run `python3 exercise.py` (about 3 seconds). After that, work out why one real marble of 20 already stops most chains from ending with one colour.
