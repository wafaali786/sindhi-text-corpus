# Step 4: Continued pretraining of XLM-R on your Sindhi corpus (Google Colab, GPU needed)
# BEFORE RUNNING: Runtime -> Change runtime type -> T4 GPU
# Each "# %%" marks a new cell: paste each block into its own cell and run in order.

# %% Cell 1: Connect Drive, set paths, install packages, check GPU
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/sindhi_corpus'
CLEAN = f'{BASE}/clean'
DATA = f'{BASE}/tokenized'
OUT = f'{BASE}/model_xlmr_sindhi'
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

!pip install -q transformers datasets accelerate
!nvidia-smi

# %% Cell 2: Tokenize the text with the XLM-R tokenizer and cut it into 256-token blocks (saved to Drive)
import itertools
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

MODEL = 'xlm-roberta-base'
BLOCK = 256            # tokens per training example (254 text tokens + 2 special tokens)
TRAIN_LINES = 400000   # use a subset of the train file so training fits in free Colab time
VAL_LINES = 2000
TEST_LINES = 10000

tok = AutoTokenizer.from_pretrained(MODEL)
tok.model_max_length = 10**6   # silences the "sequence too long" warning (we cut blocks ourselves)

def tokenize(batch):
    return tok(batch['text'], add_special_tokens=False)

def group(batch):
    ids = list(itertools.chain.from_iterable(batch['input_ids']))
    size = BLOCK - 2
    n = (len(ids) // size) * size
    blocks = [[tok.cls_token_id] + ids[i:i + size] + [tok.sep_token_id] for i in range(0, n, size)]
    return {'input_ids': blocks, 'attention_mask': [[1] * BLOCK for _ in blocks]}

def build(filename, n_lines, save_as):
    path = f'{DATA}/{save_as}'
    if os.path.exists(path):
        print('already built:', save_as)
        return load_from_disk(path)
    ds = load_dataset('text', data_files=f'{CLEAN}/{filename}', split='train')
    ds = ds.select(range(min(n_lines, len(ds))))
    ds = ds.map(tokenize, batched=True, remove_columns=['text'])
    ds = ds.map(group, batched=True, batch_size=1000, remove_columns=['input_ids', 'attention_mask'])
    ds.save_to_disk(path)
    print(save_as, ':', len(ds), 'blocks')
    return ds

train_ds = build('train.txt', TRAIN_LINES, 'train')
val_ds = build('val.txt', VAL_LINES, 'val')
test_ds = build('test.txt', TEST_LINES, 'test')

# %% Cell 3: BASELINE - how well does the ORIGINAL XLM-R handle your Sindhi test set?
import json, math, torch
from transformers import (AutoModelForMaskedLM, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)

def eval_perplexity(model, dataset):
    args = TrainingArguments(output_dir='/content/tmp_eval', per_device_eval_batch_size=8,
                             fp16=True, report_to='none')
    trainer = Trainer(model=model, args=args, data_collator=collator, eval_dataset=dataset)
    set_seed(42)   # same random masking every time, so before/after is a fair comparison
    loss = trainer.evaluate()['eval_loss']
    return loss, math.exp(loss)

base_model = AutoModelForMaskedLM.from_pretrained(MODEL)
loss, ppl = eval_perplexity(base_model, test_ds)
print(f'BASELINE  loss={loss:.4f}  perplexity={ppl:.2f}')

with open(f'{BASE}/results.json', 'w') as f:
    json.dump({'baseline_loss': loss, 'baseline_perplexity': ppl}, f)

del base_model
torch.cuda.empty_cache()

# %% Cell 4: TRAIN (the long cell). Checkpoints are saved to Drive every 500 steps.
# If Colab disconnects: reconnect, run Cells 1, 2, 3 (skip nothing), then run this cell again. It resumes by itself.
MAX_STEPS = 3000   # lower this if it is too slow (look at the speed in the progress bar)

model = AutoModelForMaskedLM.from_pretrained(MODEL)

args = TrainingArguments(
    output_dir=OUT,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,     # effective batch = 64 blocks
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=180,                  # about 6% of the 3000 steps
    weight_decay=0.01,
    fp16=True,
    eval_strategy='steps',
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    dataloader_num_workers=2,
    report_to='none',
    seed=42,
)

trainer = Trainer(model=model, args=args, data_collator=collator,
                  train_dataset=train_ds, eval_dataset=val_ds)

has_checkpoint = any(d.startswith('checkpoint-') for d in os.listdir(OUT))
trainer.train(resume_from_checkpoint=True if has_checkpoint else None)

trainer.save_model(f'{OUT}/final')
tok.save_pretrained(f'{OUT}/final')
print('Saved final model to', f'{OUT}/final')

# %% Cell 5: AFTER training - compare with the baseline (this is your main result)
final_model = AutoModelForMaskedLM.from_pretrained(f'{OUT}/final')
loss2, ppl2 = eval_perplexity(final_model, test_ds)

with open(f'{BASE}/results.json') as f:
    res = json.load(f)
res.update({'trained_loss': loss2, 'trained_perplexity': ppl2})
with open(f'{BASE}/results.json', 'w') as f:
    json.dump(res, f)

print(f"Perplexity BEFORE training: {res['baseline_perplexity']:.2f}")
print(f"Perplexity AFTER  training: {ppl2:.2f}")
