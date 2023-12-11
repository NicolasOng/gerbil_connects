import argparse
import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

parser = argparse.ArgumentParser()
parser.add_argument('gold', type=str)
parser.add_argument('pred', type=str)

args = parser.parse_args()

gold_file = args.gold
pred_file = args.pred

gold_json = read_json_file(gold_file)
pred_json = read_json_file(pred_file)

gold_annotations = sorted(gold_json, key=lambda x: x['text'])
pred_annotations = sorted(pred_json, key=lambda x: x['text'])

for i in range(len(gold_annotations)):
    gold_annotation = gold_annotations[i]
    pred_annotation = pred_annotations[i]
    assert gold_annotation['text'] == pred_annotation['text']

    gold_spans = gold_annotation['value']
    pred_spans = pred_annotation['value']

    # overgeneration