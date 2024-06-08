import collections 
import unicodedata
import requests
import json
from datasets import load_dataset


def load_runs(path, output_score=False): # support .trec file only
    run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            run_dict[qid] += [(docid, float(rank), float(score))]

    sorted_run_dict = collections.OrderedDict()
    for qid, docid_ranks in run_dict.items():
        sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False) 
        if output_score:
            sorted_run_dict[qid] = [(docid, rel_score) for docid, rel_rank, rel_score in sorted_docid_ranks]
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]

    return sorted_run_dict

def load_topic(path):
    topic = {}
    if 'miracl' in path:
        with open(path, 'r') as f:
            for line in f:
                qid, qtext = line.split('\t')
                topic[str(qid.strip())] = qtext.strip()

    if 'neuclir' in path:
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                qid = data['id'].strip()
                qtext = data['text'].strip()
                topic[str(qid)] = qtext

    return topic

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

# def get_plaid_response(
#     qid='0', 
#     query='helloworld', 
#     topk=10, 
#     lang='zho', 
#     save_results=False
# ):
#     url=f"""
#         https://trec-neuclir-search.umiacs.umd.edu/query?query='{query}'key=allInTheGroove33&content=true&limit={topk}&lang={lang}
#     """
#     response = requests.get(url.strip())
#     response = json.loads(response.content)
#
#     query_ = response['query']
#     system = response['system']
#     results = response['results']
#
#     to_return = {
#             'qid': qid, 'query': query, 
#             'docid': [], 'content': [], 'scores': []
#     }
#
#     for result in results:
#         to_return['docid'].append(result['doc_id'])
#         normalized_content = unicodedata.normalize('NFKC', result['content'])
#         to_return['content'].append(normalized_content)
#
#     if save_results:
#         with open(f'results-plaid-{qid}-{topk}-{lang}.json', 'w') as fout:
#             json.dumps(to_return, fout, ensure_ascii=False)
#     return to_return
#

# def get_bm25_response(
#     searcher, 
#     qid='0', query='helloworld', 
#     topk=10, 
#     lang='zho', 
#     save_results=False
# ):
#     hits = searcher.search(q=query, k=topk)
#     return to_return
