# Fourier Embeddings

This repository contains the experiments for the paper ["Symmetries in language statistics shape the geometry of model representations."](https://arxiv.org/abs/2602.15029v1)

Authors: Dhruva Karkada, Daniel Korchinski, Andres Nava, Matthieu Wyart, Yasaman Bahri.

## Data Preparation

- [prepare_enwiki.py](prepare_enwiki.py) - Downloads and preprocesses Wikipedia text, builds vocabulary, and creates a tokenized corpus
- [compute_cooccurrence.py](compute_cooccurrence.py) - Computes word co-occurrence statistics with configurable context windows

## Analysis Notebooks

- [fig1.ipynb](fig1.ipynb), [fig2.ipynb](fig2.ipynb), [fig3.ipynb](fig3.ipynb), [fig4.ipynb](fig4.ipynb) - Main text figures and various appendix figures
- [mstar.ipynb](mstar.ipynb) - Appendix figures for the M* co-occurrence matrix
- [lattice.ipynb](lattice.ipynb) - 2D lattice/grid embeddings
- [llm_activations_extraction.ipynb](llm_activations_extraction.ipynb) - Extraction of hidden activations from language models

## Requirements

See [pyproject.toml](pyproject.toml) for dependencies. Python 3.13+ required.
