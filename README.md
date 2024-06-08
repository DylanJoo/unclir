# UnXLIR
Unsupervised multi-lingual/cross-lingual information retrieval.

---

## Overview
In this track, we target on how embeddings behave in the context of 
(i) multilingaul search and (ii) crosslingual search.
We will investigate this with two scenarios (benchmarks): (1) multilingual passage re-ranking (MIRACL) and (2) crosslingual passage re-ranking (NeuCLIR).

### TAKS: Multi-/Cross-lingual Passage Re-ranking 
We are going the create a re-ranking models $s = F(q, d)$ to predict the relevance score of a query $q$ and a set of 1000 candidate passages $[p_1, p_2, ...p_{1000}]$.
The (initial) candidate passage retrieval could be done by an efficient mdoels such as term-matching models or dense retrieval.

Then, for each query $q$, the re-ranking model will re-order the candidate with the estimated relevance scores $s$ of every candidate passages. 
You can specify different depth (top-$k$) you would like to re-rank. There might be an efficiency-effectivenesss trade-off (deeper (larger $k$ means lower query latency, vice versa).

#### Mutlilingaul
In this scenario, the query and passage are in the same langauge (i.e., monolingual). 
We subsample the MIRACL in to Persian, Russian and Chinese (`fa`, `ru`, `zh`), to align the target langauges used in NeuCLIR.

| Dataset    | lang | retrieval | metric    |  score  | 
| ---        | ---  | ---       | ---       | ---     |
| miracl-dev | en   | bm25      | nDCG@10   | 0.3504  | 
| miracl-dev | fa   | bm25      | nDCG@10   | 0.3332  | 
| miracl-dev | ru   | bm25      | nDCG@10   | 0.3342  | 
| miracl-dev | zh   | bm25      | nDCG@10   | 0.1801  | 
| miracl-dev | en   | bm25      | Recall@10 | 0.4515  | 
| miracl-dev | fa   | bm25      | Recall@10 | 0.4368  | 
| miracl-dev | ru   | bm25      | Recall@10 | 0.3991  | 
| miracl-dev | zh   | bm25      | Recall@10 | 0.2504  | 

#### Crosslingual 
In this scenario, the query is in English (source language) and passage are in the other language (e.g., Chinese).

| Dataset    | lang | retrieval | metric    |  score  | 
| ---        | ---  | ---       | ---       | ---     |

Note that we use the google-translated query (i.e., English to target language) for the BM25 search. However, you can also try dense retrieval, which can bypassa traslation step and directly retrieve passage in target languages.

### DATA

#### Corpora
We have preprocessed the data and put it on [Huggingface](https://huggingface.co/datasets/DylanJHJ/essir-xlir). 
You can `git clone` the entire dataset repo, but make sure you have `git-lfs` installed properly (e.g., `conda install git-lfs`) 
```
git clone https://huggingface.co/datasets/DylanJHJ/essir-xlir
```
This experimental dataset are from [MIRACL](https://project-miracl.github.io/) and [NeuCLIR'23](https://neuclir.github.io/2023). 
Note that we subsample MIRACL into 3 languages: Chinese(`zho`), Russian(`rus`), and Persian(`fas`) as they are the target languages used in NeuCLIR.
*To download the raw data from scratch, please refer to [data](data).*

#### Topics (queries) and Qrels (labels)
You can find them in [data/miracl](data/miracl) and [data/neuclir](data/neuclir).

#### Runs (results) of top-1000 candidates 
You can find them in [runs](runs). 
Namining format: run.{miracl/neuclir}.{dev/translate}.bm25.{lang}.txt
The results are shown in table above.

#### (Optional) Preprocessed and Index 
As we have provided the runs (results), you won't need to do the preprocessing and indexing agiain. 
But if you want to reproduce youself, please refer to [retrieval](retrieval/readme.md).

### Baseline models
The reranking models include: (1) multilingual miniLM and (2) multilingual monoT5

#### Mutlilingaul
| Dataset    | lang | retrieval | metric    |  score  | 
| ---        | ---  | ---       | ---       | ---     |
| miracl-dev | en   | bm25      | nDCG@10   | 0.3504  | 
| miracl-dev | fa   | bm25      | nDCG@10   | 0.3332  | 
| miracl-dev | ru   | bm25      | nDCG@10   | 0.3342  | 
| miracl-dev | zh   | bm25      | nDCG@10   | 0.1801  | 
| ---        | ---  | ---       | ---       | ---     |
| miracl-dev | en   | bm25      | nDCG@10   | 0.3504  | 
| miracl-dev | fa   | bm25      | nDCG@10   | 0.3332  | 
| miracl-dev | ru   | bm25      | nDCG@10   | 0.3342  | 
| miracl-dev | zh   | bm25      | nDCG@10   | 0.1801  | 

