import numpy as np
from numba import njit
from tqdm import tqdm
import os
import sys

from FileManager import FileManager

if len(sys.argv) < 2:
    print("Usage: python compute_cooccurrence.py <corpus_dirname> <opt:vocab_sz>")
    sys.exit(1)
corpus_vocabsz = int(sys.argv[2]) if len(sys.argv) == 3 else None

dirname = sys.argv[1]
data_dir = os.path.join(os.getenv("DATASETPATH"), "enwiki", dirname)
if not os.path.exists(data_dir):
    print(f"Directory {data_dir} does not exist.")
    sys.exit(1)
data_fm = FileManager(data_dir)

context_len = 16
dtype = np.uint16
if corpus_vocabsz is None:
    corpus_vocabsz = len(data_fm.load("word_counts.pickle"))
assert corpus_vocabsz > 0
print(f"Using vocabulary size {corpus_vocabsz}")
Crwij = np.zeros((corpus_vocabsz, corpus_vocabsz), dtype=np.float32)
Cij = np.zeros((corpus_vocabsz, corpus_vocabsz), dtype=np.float32)
article_idxs = data_fm.load("article_arr_idxs.npy")
corpus_fn = data_fm.get_filename("corpus.bin")

@njit
def update_cooccurrence(Crwij, Cij, article):
    for i in range(0, len(article) - (context_len+1)):
        for j in range(1, context_len+1):
            w, v = article[i], article[i+j]
            reweight = context_len+1-j
            Crwij[w, v] += reweight
            Crwij[v, w] += reweight
            Cij[w, v] += 1
            Cij[v, w] += 1

corpus = np.memmap(corpus_fn, dtype=dtype, mode='r')
for i in tqdm(range(len(article_idxs) - 1)):
    start, stop = article_idxs[i], article_idxs[i+1] - 1
    article = corpus[start:stop]
    article = article[article < corpus_vocabsz]
    update_cooccurrence(Crwij, Cij, article)

corpus_stats = {
    "counts": Cij,
    "counts_reweight": Crwij,
    "context_len": context_len,
}
data_fm.save(corpus_stats, "corpus_stats.pickle")
