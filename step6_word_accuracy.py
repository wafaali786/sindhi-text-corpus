# Step 6: Word-guessing accuracy, original XLM-R vs your trained model (Colab, T4 GPU runtime)
# Each "# %%" marks a new cell: paste each block into its own cell and run in order.

# %% Cell 1: Connect Drive, load tokenizer and both models
from google.colab import drive
drive.mount('/content/drive')

import os, json, random, itertools, torch
BASE = '/content/drive/MyDrive/sindhi_corpus'
CLEAN = f'{BASE}/clean'
OUT = f'{BASE}/model_xlmr_sindhi'

!pip install -q transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

tok = AutoTokenizer.from_pretrained('xlm-roberta-base')
models = {
    'original': AutoModelForMaskedLM.from_pretrained('xlm-roberta-base').half().cuda().eval(),
    'trained': AutoModelForMaskedLM.from_pretrained(f'{OUT}/final').half().cuda().eval(),
}

# %% Cell 2: Build 1000 test questions (one hidden word each, same questions for both models)
# Only words that are a single token in XLM-R are used, because one <mask> can only stand for one token.
random.seed(7)
with open(f'{CLEAN}/test.txt', encoding='utf-8') as f:
    lines = [l.strip() for l in itertools.islice(f, 10000)]

items = []   # (text with <mask>, id of the true token)
for line in lines:
    words = line.split()
    if not 12 <= len(words) <= 30:
        continue
    i = random.randint(4, len(words) - 2)
    true_id = tok.convert_tokens_to_ids('▁' + words[i])
    if true_id == tok.unk_token_id:
        continue   # word is not a single token
    words[i] = '<mask>'
    text = ' '.join(words)
    ids = tok(text, truncation=True, max_length=256)['input_ids']
    if tok.mask_token_id not in ids:
        continue
    items.append((text, true_id))
    if len(items) == 1000:
        break
print('questions:', len(items))

# %% Cell 3: Score both models
def evaluate(model, items, bs=32):
    top1 = top5 = 0
    for b in range(0, len(items), bs):
        batch = items[b:b + bs]
        enc = tok([t for t, _ in batch], return_tensors='pt', padding=True,
                  truncation=True, max_length=256).to('cuda')
        with torch.no_grad():
            logits = model(**enc).logits
        is_mask = enc['input_ids'] == tok.mask_token_id
        for j, (_, true_id) in enumerate(batch):
            pos = is_mask[j].nonzero()[0].item()
            guesses = logits[j, pos].topk(5).indices.tolist()
            top1 += guesses[0] == true_id
            top5 += true_id in guesses
    return top1 / len(items), top5 / len(items)

results = {}
print(f'{"model":<12}{"top-1 accuracy":>16}{"top-5 accuracy":>16}')
for name, model in models.items():
    a1, a5 = evaluate(model, items)
    results[name] = {'top1': a1, 'top5': a5}
    print(f'{name:<12}{a1 * 100:>15.1f}%{a5 * 100:>15.1f}%')

# %% Cell 4: Save the numbers for your report
with open(f'{BASE}/results.json') as f:
    res = json.load(f)
res['word_accuracy'] = results
res['word_accuracy_questions'] = len(items)
with open(f'{BASE}/results.json', 'w') as f:
    json.dump(res, f)
print('saved')
