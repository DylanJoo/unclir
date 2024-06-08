import argparse
import json
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--topics", default=None, type=str)
    args = parser.parse_args()

    # add writers
    writers = {
            'en': open(args.topics.replace('.jsonl', '.en.jsonl'), 'w'),
            'fa': open(args.topics.replace('.jsonl', '.fa.jsonl'), 'w'),
            'ru': open(args.topics.replace('.jsonl', '.ru.jsonl'), 'w'),
            'zh': open(args.topics.replace('.jsonl', '.zh.jsonl'), 'w')
    }

    with open(args.topics, 'r') as f:

        for line in f:
            item = json.loads(line.strip())
            qid = item['topic_id']

            # en 
            topic = item['topics'][0]
            text = topic['topic_title']+" "+topic['topic_description'] 
            writers['en'].write(json.dumps({"id": qid, "text": text}, ensure_ascii=False)+'\n')

            # fa 
            topic = item['topics'][1]
            text = topic['topic_title']+" "+topic['topic_description'] 
            writers['fa'].write(json.dumps({"id": qid, "text": text}, ensure_ascii=False)+'\n')

            # ru 
            topic = item['topics'][2]
            text = topic['topic_title']+" "+topic['topic_description'] 
            writers['ru'].write(json.dumps({"id": qid, "text": text}, ensure_ascii=False)+'\n')

            # zh 
            topic = item['topics'][3]
            text = topic['topic_title']+" "+topic['topic_description'] 
            writers['zh'].write(json.dumps({"id": qid, "text": text}, ensure_ascii=False)+'\n')

    for lang, writer in writers.items():
        writer.close()
