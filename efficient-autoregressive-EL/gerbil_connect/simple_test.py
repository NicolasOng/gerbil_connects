import sys
import json

with open('./gerbil_connect/config.json', 'r') as f:
    config = json.load(f)

REPO_LOCATION = config['REPO_LOCATION']

sys.path.append(REPO_LOCATION)

from src.model.efficient_el import EfficientEL

def get_formatted_spans(preds):
    preds = preds[0]
    final_list = []
    for pred in preds:
        start = pred[0]
        end = pred[1]
        entity_list = pred[2]
        entity = entity_list[0][0]
        final_list.append((start, end, entity))
    return final_list

def print_annotated_document(doc, annotations):
    for pred in annotations:
        start = pred[0]
        end = pred[1]
        entity = pred[2]
        print(start, end, entity)
        print(doc[start - 10:start] + "*" + doc[start:end] + "*" + doc[end:end+10])
        print("")

import pickle
with open("gold_documents_new.pkl", 'rb') as file:
    new_gold_documents = pickle.load(file)

# loading the model on GPU and setting the the threshold to the
# optimal value (based on AIDA validation set)
model = EfficientEL.load_from_checkpoint("./models/model.ckpt", strict=False).eval().cuda()
model.hparams.threshold = -3.2

# loading the KB with the entities
model.generate_global_trie()

for i, item in enumerate(new_gold_documents):
    #if (i != 145): continue
    print("--------", i)
    raw_text = item["raw_text"]
    preds = model.sample([raw_text])
    print(raw_text)
    print(preds)
    if (len(preds) == 1):
        # sometimes, the model doesn't return anything.
        # happens on the document starting with "Delphis Hanover weekly municipal bond yields."
        fp = get_formatted_spans(preds)
        print_annotated_document(raw_text, fp)
