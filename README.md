# SemEval-2027 RETECO: A Shared Task on Reasoning-Oriented Retrieval

This is an individual course project for CSCI 5832 (Natural Language Processing). It targets Track 1, Sub-track 1a (Temporal Retrieval) of SemEval-2027 Task 1 (RETECO). Systems are evaluated with nDCG@10.

- Task website: https://datascienceuibk.github.io/RETECO/
- Dataset: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027
- Official starter kit: https://github.com/DataScienceUIBK/RETECO

## Data

The RETECO training and dev data is stored in `data/`. Sub-track 1a uses `data/track1_tempo/`.

The document corpora (`documents.jsonl` in each domain folder) are not committed. Most are larger than 50 MB, and GitHub rejects files over 100 MB. They are listed in `.gitignore`, so you have to download them locally. The queries, qrels, steps, guidance and manifest files are committed. To download or re-download the full dataset:

```
huggingface-cli download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset --local-dir data/
```

Newer versions of `huggingface_hub` replace `huggingface-cli` with `hf`. The equivalent command is:

```
hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset --local-dir data/
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
