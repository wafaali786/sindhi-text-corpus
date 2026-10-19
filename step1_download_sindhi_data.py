# Step 1: Download the base Sindhi data (CC-100 + Sindhi Wikipedia)
# Run this in Google Colab. Each "# %%" marks a new cell: paste each block into its own cell.

# %% Cell 1: Connect Google Drive (so files survive when Colab disconnects)
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/sindhi_corpus'
RAW = f'{BASE}/raw'
os.makedirs(RAW, exist_ok=True)
print('Saving to:', RAW)

# %% Cell 2: Install packages
!pip install -q datasets

# %% Cell 3: Download CC-100 Sindhi (web-crawled text, one sentence/line, documents separated by blank lines)
!wget -q -O {RAW}/cc100_sd.txt.xz https://data.statmt.org/cc-100/sd.txt.xz
!unxz -f {RAW}/cc100_sd.txt.xz
!ls -lh {RAW}

# %% Cell 4: Download Sindhi Wikipedia (clean article text)
from datasets import load_dataset

wiki = load_dataset('wikimedia/wikipedia', '20231101.sd', split='train')
print(wiki)

with open(f'{RAW}/wiki_sd.txt', 'w', encoding='utf-8') as f:
    for row in wiki:
        text = row['text'].strip()
        if text:
            f.write(text + '\n\n')

# %% Cell 5: Check what you got (write these numbers in your report)
def stats(path):
    lines = words = chars = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines += 1
                words += len(line.split())
                chars += len(line)
    size_mb = os.path.getsize(path) / 1e6
    print(f'{os.path.basename(path)}: {size_mb:.1f} MB | {lines:,} lines | {words:,} words')

stats(f'{RAW}/cc100_sd.txt')
stats(f'{RAW}/wiki_sd.txt')

# %% Cell 6: Look at a few samples with your own eyes (you read Sindhi, so check quality!)
for name in ['cc100_sd.txt', 'wiki_sd.txt']:
    print('=====', name)
    with open(f'{RAW}/{name}', encoding='utf-8') as f:
        shown = 0
        for line in f:
            if line.strip():
                print(line.strip()[:200])
                shown += 1
            if shown == 5:
                break
