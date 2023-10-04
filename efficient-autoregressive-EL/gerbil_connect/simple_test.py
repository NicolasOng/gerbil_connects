import sys
import json
import re

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

def sentence_tokenize(raw_text):
    '''
    splits a string into sentences.
    '''
    return sent_tokenize(raw_text)

with open('./gerbil_connect/config.json', 'r') as f:
    config = json.load(f)

REPO_LOCATION = config['REPO_LOCATION']

sys.path.append(REPO_LOCATION)

from src.model.efficient_el import EfficientEL

from gerbil_connect.helper_functions import character_to_character_index

def get_formatted_spans(preds, raw_text, dataset_text):
    final_list = []
    for pred in preds:
        start = pred[0]
        end = pred[1]
        entity_list = pred[2]
        entity = entity_list[0][0]
        # see get_markdown in src.utils
        entity = entity.replace(" ", "_")
        # convert the start/end
        raw_start = character_to_character_index(dataset_text, raw_text, start)
        raw_end = character_to_character_index(dataset_text, raw_text, end - 1) + 1
        final_list.append((raw_start, raw_end, entity))
    return final_list

def print_annotated_document(doc, annotations):
    for pred in annotations:
        start = pred[0]
        end = pred[1]
        entity = pred[2]
        print(start, end, entity)
        print(doc[start - 10:start] + "*" + doc[start:end] + "*" + doc[end:end+10])
        print("")

def load_jsonl_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def remove_whitespaces(s):
    '''
    removes whitespaces - spaces, tabs, newlines.
    '''
    return re.sub(r'\s+', '', s)

import pickle
with open("gold_documents_new.pkl", 'rb') as file:
    new_gold_documents = pickle.load(file)

# get the candidate sets
aida_test_dataset = load_jsonl_file("./data/aida_test_dataset.jsonl")
aida_val_dataset = load_jsonl_file("./data/aida_val_dataset.jsonl")
aida_dataset = aida_test_dataset + aida_val_dataset
for i, item in enumerate(aida_dataset):
    item["input_no_white_space"] = remove_whitespaces(item["input"])

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

    raw_text_no_whitespace = remove_whitespaces(raw_text)
    for i, item in enumerate(aida_dataset):
        if item["input_no_white_space"] == raw_text_no_whitespace:
            dataset_id = item["id"]
            dataset_input = item["input"]
            dataset_candidates = item["candidates"]
            dataset_anchors = item["anchors"]
            break
    
    model_preds = model.sample([dataset_input], candidates=[dataset_candidates], anchors=[dataset_anchors], all_targets=True)
    #print(raw_text)
    #print(model_preds)
    if (len(model_preds) == 1):
        # sometimes, the model doesn't return anything.
        # happens on the document starting with "Delphis Hanover weekly municipal bond yields."
        fp = get_formatted_spans(model_preds[0], raw_text, dataset_input)
        print_annotated_document(raw_text, fp)
    break
    