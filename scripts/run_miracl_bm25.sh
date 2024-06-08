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
dataset=miracl

for lang in en fa ru zh;do

    mkdir -p data/${lang}
    python retrieval/bm25_search.py \
        --k 100 --k1 0.9 --b 0.4 \
        --index ${index_dir}/${dataset}/${dataset}-${lang}.lucene \
        --topic data/topics/topics.miracl-v1.0-${lang}-dev.tsv \
        --lang ${lang} \
        --batch_size 32 \
        --output runs/run.${dataset}.dev.bm25.${lang}.txt
done

