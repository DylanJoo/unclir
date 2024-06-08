# UnXLIR
---
Unsupervised multi-lingual/cross-lingual information retrieval.

---
## Overview
In this track, we target on how embeddings behave in the context of 
(i) multilingaul search and (ii) crosslingual search.

### Goal 
### The reranking tasks 

#### The first-stage candidates
- Mutlilingaul 
| Dataset    | lang | retrieval | metric    |  score  | 
| ---        | ---  | ---       | ---       | ---     |
| miracl-dev | en   | bm25      |  nDCG@10  | 0.3504  | 
| miracl-dev | fa   | bm25      |  nDCG@10  | 0.3332  | 
| miracl-dev | ru   | bm25      |  nDCG@10  | 0.3342  | 
| miracl-dev | zh   | bm25      |  nDCG@10  | 0.1801  | 
| miracl-dev | en   | bm25      | Recall@10 | 0.4515  | 
| miracl-dev | fa   | bm25      | Recall@10 | 0.4368  | 
| miracl-dev | ru   | bm25      | Recall@10 | 0.3991  | 
| miracl-dev | zh   | bm25      | Recall@10 | 0.2504  | 

- Crosslingual 
| Dataset    | lang | retrieval | metric    |  score  | 
| ---        | ---  | ---       | ---       | ---     |

## Preparation: before start

### Datasets
The experimental datasets we used are from [MIRACL](https://project-miracl.github.io/) and [NeuCLIR'23](https://neuclir.github.io/2023)
We will only use the languages: Chinese(`zho`), Russian(`rus`), and Persian(`fas`) in MIRACL as we aim to align it to NeuCLIR.

We have preprocessed the data and put it on [Huggingface](https://huggingface.co/datasets/DylanJHJ/essir-xlir).
<li>To download the raw data from scratch, please refer to [data](data).</li>

After download the data, we recommend to use the symbolic link to refer the downloaded corpus.
Especially you want to reproduce BM25 (it will take all the file in folder for indexing, so it's better to separate them)
```
DATA_DIR=<your-save-path>

for dataset in miracl neuclir;do
    for lang in zh ru fa;do
        mkdir data/${dataset}/${lang}/
        ln -s ${DATA_DIR}/${dataset}-${lang}.jsonl data/${dataset}/${lang}/
    done
done
```

## The multi-stage pipeline 
### Candidate retrieval
We have prepared all the top-1000 retrieval results [here]().

For examples, the result of NeuCLIR'23 Englisht-to-Chinese look like:
```
TBD
```
The naminig format is: {miracl/neuclir}-{lang}-{retriever}.run

To reproduce first-stage retrieval, please refer to [retrieval](retrieval). You can also try this yourself! 
Multilingual DR and LSR methods are very active research area. E.g., PLAIDx, mDPR, ...

### Models
This track is focusing on reranking task. The reranking models include:
(1) multilingual miniLM
(2) multilingual monoT5
