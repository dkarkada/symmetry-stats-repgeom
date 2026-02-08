import numpy as np
from tqdm import tqdm
import sys
import os
import re
import pickle
from collections import Counter

from datasets import load_dataset, load_from_disk
from FileManager import FileManager


class Vocabulary:

    def __init__(self, word_counts):
        self.words = np.array([word for word, c in word_counts])
        self.counts = np.array([c for word, c in word_counts])
        self.word2token = {word:tok for tok, word in enumerate(self.words)}
        self.size = len(self.words)

    def get_count(self, word):
        if word not in self.word2token:
            return 0
        return self.counts[self.word2token.get(word)]

    def to_words(self, tokens):
        return " ".join([self.words[tok] for tok in tokens])


# Full wikipedia is 6407814 articles
# In 200k articles: mean article length is 678, max is 50k
#                   [0.5, 0.9, 0.95, 0.99] quantiles = [364, 1470, 2233, 5362]

vocab_sz = 65535    # must be no more than 65535
min_length = 200    # creates 3.37M articles
dirname = "min200"
substitutes = {}
if os.path.isfile("substitutions.pickle"):
    with open("substitutions.pickle", "rb") as f:
        substitutes = pickle.load(f)
print(f"Number of substitutions: {len(substitutes)}")

data_dir = os.path.join(os.getenv("DATASETPATH"), "enwiki")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_fm = FileManager(data_dir)

try:
    cleaned_ds = load_from_disk(data_fm.get_filename("tokenized-integers"))
    print("Found cleaned dataset.", flush=True)
except:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    train_ds = ds["train"]

    print("Cleaning dataset... ", flush=True)
    
    def preprocess_text(article):
        # lowercase
        text = article["text"].lower()
        # handle numeric commas: remove them entirely
        text = re.sub(r'\d+,\d+', '', text)
        # handle decimal numbers: remove them entirely
        text = re.sub(r'\d+\.\d+', '', text)
        # handle apostrophes between letters: remove without space
        text = re.sub(r'(?<=[a-z])\'(?=[a-z])', '', text)
        # replace all non-alphanumeric chars with spaces
        text = re.sub(r'[^a-z0-9]+', ' ', text)
        # apply substitutions
        for k, v in substitutes.items():
            pattern = rf'\b{k}\b'
            text = re.sub(pattern, v, text)
        # remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return {"text": text.strip().split()}

    cleaned_ds = train_ds.map(preprocess_text, num_proc=48)
    cleaned_ds.save_to_disk(data_fm.get_filename("tokenized-integers"))
    print("done.", flush=True)

data_fm.set_filepath(dirname)
# Collect unigram statistics, construct vocabulary
counter = Counter()
article_idxs = []
arr_len_upperbound = 0
is_logging = not sys.stderr.isatty()
print("Collecting vocabulary statistics... ", flush=True)
for i in tqdm(range(len(cleaned_ds)), disable=is_logging):
    article = cleaned_ds[i]["text"]
    if len(article) >= min_length:
        article_idxs.append(i)
        arr_len_upperbound += 1 + len(article)
        counter.update(article)
counter = {word: c for word, c in counter.items()}
print("done.", flush=True)

word_counts = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
word_counts = word_counts[:vocab_sz]
data_fm.save(word_counts, "word_counts.pickle")
vocab = Vocabulary(word_counts)

# Construct bin file
print(f"Using {100*len(article_idxs)/6407814:0.2f}% of articles, min length {min_length}.", flush=True)
print(f"Creating bin file (vocab size = {vocab_sz})... ", flush=True)
filename = data_fm.get_filename("corpus.bin")
arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len_upperbound,))
EOF_token = vocab_sz
assert EOF_token >= len(vocab.words)
assert EOF_token < 2**16
idx = 0
article_arr_idxs = []
for i in tqdm(article_idxs, disable=is_logging):
    article = cleaned_ds[i]["text"]
    assert len(article) >= min_length
    corpus = np.array([vocab.word2token[word] for word in article
                       if word in vocab.word2token] + [EOF_token],
                      dtype=np.uint16)
    assert idx + len(corpus) <= arr_len_upperbound
    arr[idx : idx + len(corpus)] = corpus
    article_arr_idxs.append(idx)
    idx += len(corpus)
article_arr_idxs.append(idx)
arr.flush()
with open(filename, 'r+b') as f:
    f.truncate(idx * np.dtype(np.uint16).itemsize)
print(f"\t Number of articles: {len(article_idxs)}", flush=True)
print(f"\t Number of tokens: {idx}", flush=True)
data_fm.save(np.array(article_arr_idxs), "article_arr_idxs.npy")

print("done.")
