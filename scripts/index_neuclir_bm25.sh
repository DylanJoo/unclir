#!/bin/sh
#SBATCH --job-name=bm25
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Indexing step please check retrieval/bm25/readme.md.
# Start the experiment.
index_dir=${HOME}/indexes/essir-xlir
data_dir=${HOME}/datasets/essir-xlir
dataset=neuclir

mkdir -p data/temp

for lang in fa ru zh;do

    # add soft link to a standalone folder
    ln -s ${data_dir}/neuclir/neuclir-${lang}.pyserini.jsonl data/temp/doc.jsonl
    python3 -m pyserini.index.lucene \
        --collection JsonCollection \
        --input data/temp \
        --index ${index_dir}/neuclir/neuclir-${lang}.lucene \
        --language ${lang} \
        --generator DefaultLuceneDocumentGenerator \
        --threads 9 \

    rm -rf data/temp
done
echo done

