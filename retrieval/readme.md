## Sparse retrieval (bm25) baseline  
---
In this document, we provide the procedure for sparse retrieval baseline.
We use pyserini toolkit and its implemeneted indexing/searching infrastructure. 

### MIRACL 
- Indexing: we directly downalod from [Pyserini prebuilt indexes](https://github.com/castorini/pyserini/blob/master/docs/experiments-miracl-v1.0.md).
- Searching with BM25:
Once the lucene index is ready, you can run the retrieval with the following codes. 
Please refer to [bm25_search.py](bm25_search.py) for details.

```
index_dir=${HOME}/indexes/essir-xlir
data_dir=${HOME}/datasets/essir-xlir
dataset=miracl

cd unclir/
for lang in en fa ru zh;do
    python retrieval/bm25_search.py \
        --k 100 --k1 0.9 --b 0.4 \
        --index ${index_dir}/${dataset}/${dataset}-${lang}.lucene \
        --topic data/miracl/topics/topics.miracl-v1.0-${lang}-dev.tsv \
        --lang ${lang} \
        --batch_size 32 \
        --output runs/run.${dataset}.dev.bm25.${lang}.txt
done
```

### NeuCLIR

- Preprocessing corpora:
First, we need to transform the corpus into the `.jsonl` format, to make it compatible with pyserini's API. 
You can run the code below for this convert.
```
cd unclir/
python tools/convert_neuclir_to_jsonl.py \
  --collection-path <data-dir>/neuclir-zh.jsonl 

# Then,You will find the converted file in the same folder of <data-dir>.
```

- Indexing: we use Pyserini API
```
index_dir=${HOME}/indexes/essir-xlir

cd unclir/
for lang in fa ru zh;do
    python3 -m pyserini.index.lucene \
        --collection JsonCollection \
        --input data/${lang} \
        --index ${index_dir}/neuclir/neuclir-${lang}.lucene  \
        --language ${lang} \
        --generator DefaultLuceneDocumentGenerator \
        --threads 9 \
done
```

- Searching: you can run the retrieval once the lucene indexing is done.
```
index_dir=${HOME}/indexes/essir-xlir
data_dir=${HOME}/datasets/essir-xlir
dataset=neuclir

cd unclir/
for lang in fa ru zh;do
    python retrieval/bm25_search.py \
        --k 100 --k1 0.9 --b 0.4 \
        --index ${index_dir}/${dataset}/${dataset}-${lang}.lucene \
        --topic data/neuclir/topics/neuclir-2024-topics.0605.${lang}.jsonl \
        --lang ${lang} \
        --batch_size 32 \
        --output runs/run.${dataset}.dev.bm25.${lang}.txt
done
```
