# Step 7: Upload the Sindhi corpus to Hugging Face (Google Colab, CPU runtime is fine)
# Before running: 1) make a Hugging Face account  2) make a token with WRITE access:
#   huggingface.co -> your profile picture -> Settings -> Access Tokens -> Create new token -> type "Write"
# Each "# %%" marks a new cell: paste each block into its own cell and run in order.

# %% Cell 1: Connect Drive, install, log in
from google.colab import drive
drive.mount('/content/drive')

!pip install -q huggingface_hub

from huggingface_hub import login
login()   # a box appears: paste your WRITE token there

# %% Cell 2: Settings (change USERNAME to your Hugging Face username)
USERNAME = 'USERNAME'
REPO = f'{USERNAME}/sindhi-text-corpus'
CLEAN = '/content/drive/MyDrive/sindhi_corpus/clean'

# %% Cell 3: Upload. The repo starts PRIVATE so you can check it before making it public.
# Note: val.txt is uploaded as validation.txt so Hugging Face detects the splits automatically.
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)

for local, remote in [('train.txt', 'train.txt'),
                      ('val.txt', 'validation.txt'),
                      ('test.txt', 'test.txt')]:
    print('uploading', remote, '...')
    api.upload_file(path_or_fileobj=f'{CLEAN}/{local}', path_in_repo=remote,
                    repo_id=REPO, repo_type='dataset')

# README.md: upload it from your computer in the web page (see the steps), or put it in Drive and use:
# api.upload_file(path_or_fileobj='/content/drive/MyDrive/sindhi_corpus/README.md',
#                 path_in_repo='README.md', repo_id=REPO, repo_type='dataset')
print('done: https://huggingface.co/datasets/' + REPO)

# %% Cell 4: Check that it loads the way other people will load it
!pip install -q datasets
from datasets import load_dataset
ds = load_dataset(REPO)          # works for a private repo because you are logged in
print(ds)
print(ds['train'][0])

# %% Cell 5: When you are happy, make it public
api.update_repo_settings(REPO, repo_type='dataset', private=False)
print('now public')
