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
conda activate exa-dm_env

# Indexing step please check retrieval/bm25/readme.md.
# Start the experiment.

for lang in en fa ru zh;do
    echo -ne "miracl-dev | ${lang} | bm25 | nDCG@10   | "
    ~/trec_eval-9.0.7/trec_eval -c -m ndcg_cut.10 \
        data/miracl/qrels/qrels.miracl-v1.0-${lang}-dev.tsv \
        runs/run.miracl.dev.bm25.${lang}.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done

for lang in en fa ru zh;do
    echo -ne "miracl-dev | ${lang} | bm25 | Recall@10 | "
    ~/trec_eval-9.0.7/trec_eval -c -m recall.10 \
        data/miracl/qrels/qrels.miracl-v1.0-${lang}-dev.tsv \
        runs/run.miracl.dev.bm25.${lang}.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
