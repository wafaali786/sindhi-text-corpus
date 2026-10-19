# Step 3: Split the corpus (train/val/test) and train Sindhi tokenizers
# Google Colab, normal CPU runtime (no GPU needed yet).
# Each "# %%" marks a new cell: paste each block into its own cell and run in order.

# %% Cell 1: Connect Drive, set paths, install packages
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/sindhi_corpus'
CLEAN = f'{BASE}/clean'
TOK = f'{BASE}/tokenizers'
os.makedirs(TOK, exist_ok=True)
print(os.listdir(CLEAN))   # should show sindhi_clean.txt

!pip install -q sentencepiece transformers

# %% Cell 2: Split into train / validation / test (shuffled, fixed seed so it can be repeated)
import random
random.seed(42)

with open(f'{CLEAN}/sindhi_clean.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
random.shuffle(lines)

N_HOLDOUT = 10000
val = lines[:N_HOLDOUT]
test = lines[N_HOLDOUT:2 * N_HOLDOUT]
train = lines[2 * N_HOLDOUT:]

for name, data in [('train', train), ('val', val), ('test', test)]:
    with open(f'{CLEAN}/{name}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(data) + '\n')
    words = sum(len(l.split()) for l in data)
    print(f'{name}: {len(data):,} lines | {words:,} words')

del lines, train, val, test   # free RAM

# %% Cell 3: Train tokenizers (this is the slow cell: roughly 5-15 min per tokenizer)
# Trim the list if you are short on time, e.g. keep only the two 16000 ones.
import sentencepiece as spm, time

configs = [('bpe', 16000), ('unigram', 16000), ('bpe', 32000), ('unigram', 32000)]

for model_type, vocab_size in configs:
    prefix = f'{TOK}/sd_{model_type}_{vocab_size}'
    if os.path.exists(prefix + '.model'):
        print('already done:', prefix)
        continue
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=f'{CLEAN}/train.txt',
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        input_sentence_size=300000,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
        num_threads=2,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    )
    print(f'{model_type} {vocab_size}: done in {(time.time() - t0) / 60:.1f} min')

# %% Cell 4: Compare tokenizers (fewer tokens per word = better fit for Sindhi)
import itertools
from transformers import AutoTokenizer

with open(f'{CLEAN}/val.txt', encoding='utf-8') as f:
    sample = [l.strip() for l in itertools.islice(f, 3000)]
n_words = sum(len(s.split()) for s in sample)

print(f'{"tokenizer":<24}{"tokens per word":>16}')
for model_type, vocab_size in configs:
    sp = spm.SentencePieceProcessor(model_file=f'{TOK}/sd_{model_type}_{vocab_size}.model')
    n_tok = sum(len(sp.encode(s)) for s in sample)
    print(f'{model_type + " " + str(vocab_size):<24}{n_tok / n_words:>16.2f}')

xlmr = AutoTokenizer.from_pretrained('xlm-roberta-base')
n_tok = sum(len(xlmr.tokenize(s)) for s in sample)
print(f'{"XLM-R (original)":<24}{n_tok / n_words:>16.2f}')

# %% Cell 5: See how one sentence gets split
sp = spm.SentencePieceProcessor(model_file=f'{TOK}/sd_unigram_32000.model')
print(sp.encode(sample[0][:150], out_type=str))
