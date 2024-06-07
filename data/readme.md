# Collection (documents)

---
The datasets for ESSIR hackathon are from MIRACL and NeuCLIR.

<!-- However, we have done some downsampling to streamline the experiment overhead of compute and time.  -->
<!-- However, we subsample the corpus to make it smaller for our re-ranking task.  -->
<!-- Research preprocessing and other time costs of training/inference time. -->
---

### MIRACL 2023
#### Collection
We downloaded it from [MIRACL's huggingface](https://huggingface.co/datasets/neuclir/neuclir1/tree/main/data).
We use the Huggingface's dataset API to access the raw data. 
```python
from huggingface_hub import hf_hub_download

for lang in ['fa', 'ru', 'zh', 'en']:
	hf_hub_download('miracl/miracl', 
            filename=f'topics.miracl-v1.0-{lang}-dev.tsv', 
            subfolder=f'miracl-v1.0-{lang}/topics', 
            repo_type='dataset', 
            cache_dir='/home/dju/datasets/essir-xlir/',
            force_filename=f'topics.miracl-v1.0-{lang}-test-a.tsv')
    corpus.to_json(f"miracl-{lang}.jsonl")
```
#### Prebuilt indexes
We directly took the prebuilt sparse indexes from [Pyserini](#)
You can the prebuilt sparse indexes from Pyserini! Download [here](https://github.com/castorini/pyserini/blob/master/docs/experiments-miracl-v1.0.md)


#### Topics and Qrels

### NeuCLIR 2023
#### Collection 
We downloaded the datasets from [NeuCLIR's huggingface](https://huggingface.co/datasets/neuclir/neuclir1/tree/main/data).
```
# Note that we change the langauge codes and match them with MIRACL's (i.e., "rus" --> "ru")

wget https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/fas-00000-of-00001.jsonl.gz -O neuclir-fa.jsonl.gz 
wget https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/rus-00000-of-00001.jsonl.gz -O neuclir-ru.jsonl.gz 
wget https://huggingface.co/datasets/neuclir/neuclir1/resolve/main/data/zho-00000-of-00001.jsonl.gz -O neuclir-zh.jsonl.gz 
gunzip neuclir-fa.jsonl.gz
gunzip neuclir-ru.jsonl.gz
gunzip neuclir-zh.jsonl.gz
```
#### Topics and Qrels
The topics and qrels may not be able to share as this is particularly for TREC participants and research purpose.
Please contact TREC NeuCLIR organizers for this matter.

