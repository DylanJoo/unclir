#!/bin/sh
#SBATCH --job-name=sr
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc

# Indexing step please check retrieval/bm25/readme.md.
# Start the experiment.
for dataset in miracl neuclir;do
    index_dir=${HOME}/indexes/miracl
    data_dir=${HOME}/datasets/miracl

    # Search
    python retrieval/bm25/search.py \
        --k 100 --k1 0.9 --b 0.4 \
        --index ${index_dir}/${dataset}/bm25.lucene \
        --topic ${data_dir}/${dataset}/queries.dev.jsonl \
        --batch_size 32 \
        --output runs/run.${dataset}.dev.bm25.${lang}.txt

    # Eval
    echo -ne "beir-${dataset}  | bm25 | ${dataset} | reproduced | "
    ~/trec_eval-9.0.7/trec_eval -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/bm25/run.beir.${dataset}.bm25-multifield.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
