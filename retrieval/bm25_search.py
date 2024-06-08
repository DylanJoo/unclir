import os
import json
import argparse
from tqdm import tqdm 
from tools import load_topic, batch_iterator
from pyserini.search.lucene import LuceneSearcher

def search(args):

    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)
    searcher.set_language(args.lang)

    topics = load_topic(args.topic)
    qids = list(topics.keys())
    qtexts = list(topics.values())
    output = open(args.output, 'w')

    for (start, end) in tqdm(
            batch_iterator(range(0, len(qids)), args.batch_size, True),
            total=(len(qids)//args.batch_size)+1
    ):
        qids_batch = qids[start: end]
        qtexts_batch = qtexts[start: end]
        hits = searcher.batch_search(
                queries=qtexts_batch, 
                qids=qids_batch, 
                threads=32,
                k=args.k,
        )

        for key, value in hits.items():
            for i in range(len(hits[key])):
                output.write(f'{key} Q0 {hits[key][i].docid:4} {i+1} {hits[key][i].score:.5f} bm25\n')

    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--k1",type=float, default=4.68) # 0.5 # 0.82
    parser.add_argument("--b", type=float, default=0.87) # 0.3 # 0.68
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--topic", default=None, type=str)
    parser.add_argument("--lang", default=None, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    search(args)
    print("Done")
