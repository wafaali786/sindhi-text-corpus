# Step 5: Mask-filling demo, original XLM-R vs your trained model (Colab, T4 GPU runtime)
# Each "# %%" marks a new cell: paste each block into its own cell and run in order.

# %% Cell 1: Connect Drive, set paths, load both models
from google.colab import drive
drive.mount('/content/drive')

import os, itertools, random
BASE = '/content/drive/MyDrive/sindhi_corpus'
CLEAN = f'{BASE}/clean'
OUT = f'{BASE}/model_xlmr_sindhi'

!pip install -q transformers

from transformers import pipeline
base = pipeline('fill-mask', model='xlm-roberta-base', device=0)
tuned = pipeline('fill-mask', model=f'{OUT}/final', device=0)

# %% Cell 2: 8 test sentences with one word hidden: the models guess the missing word
random.seed(1)
with open(f'{CLEAN}/test.txt', encoding='utf-8') as f:
    lines = [l.strip() for l in itertools.islice(f, 2000)]

shown = 0
for line in lines:
    words = line.split()
    if not 12 <= len(words) <= 30:
        continue
    i = random.randint(4, len(words) - 2)
    true_word = words[i]
    words[i] = '<mask>'
    text = ' '.join(words)
    print('SENTENCE:', text)
    print('TRUE WORD:', true_word)
    for name, p in [('original', base), ('trained ', tuned)]:
        guesses = [r['token_str'] for r in p(text, top_k=3)]
        print(f'  {name}:', guesses)
    print()
    shown += 1
    if shown == 8:
        break

# %% Cell 3: Try your own sentence (replace the sentence, keep <mask> where the word is missing)
my_sentence = 'ڪراچي سنڌ جو سڀ کان وڏو <mask> آهي.'
for name, p in [('original', base), ('trained ', tuned)]:
    print(name, [r['token_str'] for r in p(my_sentence, top_k=5)])
