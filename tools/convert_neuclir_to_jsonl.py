import re
import json
import os
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert CC files jsonl to pyserini's format")
    parser.add_argument('--collection-path', required=True,)
    args = parser.parse_args()

    # add writers
    writer = open(args.collection_path.replace('.jsonl', '.pyserini.jsonl'), 'w')

    with open(args.collection_path, encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            doc_dict = json.loads(line.strip())

            # components in a dict
            doc_id = doc_dict['id']
            doc_title = re.sub('\s+', ' ', doc_dict['title'])
            doc_content = re.sub('\s+', ' ', doc_dict['text'])
            doc_text = f"{doc_title} {doc_content}"
            output_dict = {'id': doc_id, 'contents': doc_text}
            writer.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

    writer.close()
    print('Done!')
