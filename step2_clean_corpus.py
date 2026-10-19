# Step 2: Clean the Sindhi corpus (Google Colab, CPU runtime is fine)
# Each "# %%" marks a new cell: paste each block into its own cell and run in order.

# %% Cell 1: Connect Drive and set paths (same folder as Step 1)
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/sindhi_corpus'
RAW = f'{BASE}/raw'
CLEAN = f'{BASE}/clean'
os.makedirs(CLEAN, exist_ok=True)
print(os.listdir(RAW))   # should show cc100_sd.txt and wiki_sd.txt

# %% Cell 2: Cleaning rules
import re, unicodedata, hashlib
from collections import Counter

URL = re.compile(r'https?://\S+|www\.\S+|\S+@\S+')
INVISIBLE = re.compile('[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff\u0640]')  # zero-width marks + tatweel
TRAIL_LATIN = re.compile(r'\s*[–—-]\s*[A-Za-z][A-Za-z0-9\s\-\.]*$')                 # e.g. "– Daily Hilal-e-Pakistan"
ARABIC = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')
LATIN = re.compile(r'[A-Za-z]')

MIN_WORDS = 4          # drop very short lines (headings, "طب", "حوالا")
MIN_SCRIPT_RATIO = 0.8 # at least 80% of letters must be Arabic-script

def clean_line(line, stats):
    line = unicodedata.normalize('NFKC', line)
    line = URL.sub(' ', line)
    line = INVISIBLE.sub('', line)
    line = TRAIL_LATIN.sub('', line)
    line = ' '.join(line.split())
    if not line:
        stats['empty'] += 1
        return None
    if len(line.split()) < MIN_WORDS:
        stats['too_short'] += 1
        return None
    ar, lat = len(ARABIC.findall(line)), len(LATIN.findall(line))
    if ar + lat == 0 or ar / (ar + lat) < MIN_SCRIPT_RATIO:
        stats['not_sindhi_script'] += 1
        return None
    return line

# %% Cell 3: Run the cleaning (reads both files, removes duplicates, writes one clean file)
out_path = f'{CLEAN}/sindhi_clean.txt'
seen = set()
stats = Counter()

with open(out_path, 'w', encoding='utf-8') as out:
    for name in ['wiki_sd.txt', 'cc100_sd.txt']:
        with open(f'{RAW}/{name}', encoding='utf-8') as f:
            for raw in f:
                stats[f'read_{name}'] += 1
                line = clean_line(raw, stats)
                if line is None:
                    continue
                h = hashlib.md5(line.encode('utf-8')).digest()[:8]
                if h in seen:
                    stats['duplicate'] += 1
                    continue
                seen.add(h)
                out.write(line + '\n')
                stats[f'kept_{name}'] += 1

for k, v in stats.items():
    print(f'{k}: {v:,}')

# %% Cell 4: Final stats (write these numbers in your report)
lines = words = 0
with open(out_path, encoding='utf-8') as f:
    for line in f:
        lines += 1
        words += len(line.split())
print(f'sindhi_clean.txt: {os.path.getsize(out_path)/1e6:.1f} MB | {lines:,} lines | {words:,} words')

# %% Cell 5: Look at samples again (check with your own eyes)
with open(out_path, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i % 50000 == 0:
            print(line.strip()[:200])
        if i > 250000:
            break
