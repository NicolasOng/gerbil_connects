"""
This file provides a template server in which you can import your implemented entity linking model and evaluate it using
GERBIL. All the communication parts are worked out you will only need to replace `mock_entity_linking_model` with your
model and start load up its required resources in `generic_annotate` method.

This template supports both Wikipedia in-domain test sets (i.e. AIDA-CoNLL) and out-of-domain test sets (e.g. KORE).
At the end you will also have the annotation results stored in `annotate_{annotator_name}_result.json`.
"""
import os
import json
import pathlib
from threading import Lock
from flask import Flask, request
from flask_cors import CORS, cross_origin
from gerbil_connect.nif_parser import NIFParser

import re
import sys
import argparse

with open('./gerbil_connect/config.json', 'r') as f:
    config = json.load(f)

REPO_LOCATION = config['REPO_LOCATION']

sys.path.append(REPO_LOCATION)

from src.model.efficient_el import EfficientEL
from gerbil_connect.helper_functions import character_to_character_index

def ea_el_get_formatted_spans(preds, raw_text, dataset_text):
    final_list = []
    for pred in preds:
        start = pred[0]
        end = pred[1]
        entity_list = pred[2]
        entity = entity_list[0][0]

        if end <= 0:
            # sometimes, 0, 0, NIL is predicted.
            continue
    
        # see get_markdown in src.utils
        entity = entity.replace(" ", "_")
        # convert the start/end
        raw_start = character_to_character_index(dataset_text, raw_text, start)
        raw_end = character_to_character_index(dataset_text, raw_text, end - 1) + 1
        final_list.append((raw_start, raw_end, entity))
    return final_list

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

app = Flask(__name__, static_url_path='', static_folder='../../../frontend/build')
cors = CORS(app, resources={r"/suggest": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

app.add_url_rule('/', 'root', lambda: app.send_static_file('index.html'))

gerbil_communication_done = False
gerbil_query_count = 0
annotator = None
annotate_result = []
candidates_manager_to_use = None
n3_entity_to_kb_mappings = None

lock = Lock()

def get_n3_entity_to_kb_mappings():
    kb_file = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "resources" / "data" / "n3_kb_mappings.json"
    knowledge_base = json.load(kb_file.open("r"))
    return knowledge_base


def extract_dump_res_json(parsed_collection):
    return {
        "text": parsed_collection.contexts[0].mention,
        "value": [{"start": phrase.beginIndex, "end": phrase.endIndex, "tag": phrase.taIdentRef}
                  for phrase in parsed_collection.contexts[0]._context.phrases]
    }

def ea_el_model(raw_text):
    raw_text_no_whitespace = remove_whitespaces(raw_text)
    dataset_id = None
    dataset_input = raw_text
    for i, item in enumerate(aida_dataset):
        if item["input_no_white_space"] == raw_text_no_whitespace:
            dataset_id = item["id"]
            dataset_input = item["input"]
            dataset_candidates = item["candidates"]
            dataset_anchors = item["anchors"]
            break
    #candidates = [candidate for candidate_set in candidate_sets for candidate in candidate_set]
    
    # get the model predictions
    # the model can handle all the documents GERBIL sends,
    # so I won't bother with splitting into sentences.
    if args.no_candidate_sets:
        model_preds = model.sample([dataset_input])
    elif args.full_candidate_sets and (dataset_id is not None):
        # before: a list of candidate lists like: ['Culture of Japan', 'Emperor of Japan', 'Japan Japan', ...]
        # after: each candidate list is the full candidate list.
        full_dataset_candidates = [full_candidates_list.copy() for _ in dataset_candidates]
        model_preds = model.sample([dataset_input], candidates=[full_dataset_candidates], anchors=[dataset_anchors], all_targets=True)
    elif dataset_id is not None:
        model_preds = model.sample([dataset_input], candidates=[dataset_candidates], anchors=[dataset_anchors], all_targets=True)
    else:
        model_preds = model.sample([dataset_input])

    # format the model output to GEBRIL's format.
    final_preds = []
    if (len(model_preds) == 1):
        # sometimes, the model doesn't return anything.
        # happens on the document starting with "Delphis Hanover weekly municipal bond yields."
        final_preds = ea_el_get_formatted_spans(model_preds[0], raw_text, dataset_input)
    
    return final_preds

class GerbilAnnotator:
    """
    The annotator class must implement a function with the following signature
    """
    def annotate(self, nif_collection, **kwargs):
        assert len(nif_collection.contexts) == 1
        context = nif_collection.contexts[0]
        raw_text = context.mention
        # TODO We assume Wikipedia as the knowledge base, but you can consider any other knowledge base in here:
        kb_prefix = "https://en.wikipedia.org/wiki/"
        for annotation in ea_el_model(raw_text):
            # TODO you can have the replacement for mock_entity_linking_model to return the prediction_prob as well:
            prediction_probability = 1.0
            context.add_phrase(beginIndex=annotation[0], endIndex=annotation[1], score=prediction_probability,
                               annotator='http://sfu.ca/spel/gerbil_connect', taIdentRef=kb_prefix+annotation[2])

def generic_annotate(nif_bytes, kb_prefix):
    global gerbil_communication_done, gerbil_query_count, annotator, candidates_manager_to_use
    parsed_collection = NIFParser(nif_bytes.decode('utf-8').replace('\\@', '@'), format='turtle')
    if gerbil_communication_done:
        gerbil_query_count += 1
        print("Received query number {} from gerbil!".format(gerbil_query_count))
        with lock:
            annotator.annotate(parsed_collection, kb_prefix=kb_prefix, candidates_manager=candidates_manager_to_use)
    else:
        print(" * Handshake to Gerbil was successful!")
        # TODO instanciate the annotator class and load up its required resources:
        annotator = GerbilAnnotator()
        gerbil_communication_done = True
        return nif_bytes
    try:
        res = parsed_collection.nif_str(format='turtle')
        res_json = extract_dump_res_json(parsed_collection)
        annotate_result.append(res_json)
        return res
    except Exception:
        return nif_bytes

@app.route('/annotate_aida', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origins='*')
def annotate_aida():
    """Use this API for AIDA dataset."""
    return generic_annotate(request.data, "http://en.wikipedia.org/wiki/")

@app.route('/annotate_wiki', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origins='*')
def annotate_wiki():
    """Use this API for MSNBC dataset."""
    return generic_annotate(request.data, "http://en.wikipedia.org/wiki/")

@app.route('/annotate_dbpedia', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origins='*')
def annotate_dbpedia():
    """Use this API for OKE, KORE, and Derczynski datasets."""
    return generic_annotate(request.data, "http://dbpedia.org/resource/")

@app.route('/annotate_n3', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origins='*')
def annotate_n3():
    """Use this API for N3 Evaluation dataset."""
    global n3_entity_to_kb_mappings
    if n3_entity_to_kb_mappings is None:
        n3_entity_to_kb_mappings = get_n3_entity_to_kb_mappings()
    return generic_annotate(request.data, n3_entity_to_kb_mappings)

# start

parser = argparse.ArgumentParser()
parser.add_argument("--no-candidate-sets", action="store_true")
parser.add_argument("--full-candidate-sets", action="store_true")
args = parser.parse_args()

# get the candidate sets
aida_test_dataset = load_jsonl_file("./data/aida_test_dataset.jsonl")
aida_val_dataset = load_jsonl_file("./data/aida_val_dataset.jsonl")
aida_c_dataset = load_jsonl_file("./data/aida_c_dataset.jsonl")
aida_dataset = aida_test_dataset + aida_val_dataset + aida_c_dataset
for i, item in enumerate(aida_dataset):
    item["input_no_white_space"] = remove_whitespaces(item["input"])

# loading the model on GPU and setting the the threshold to the
# optimal value (based on AIDA validation set)
model = EfficientEL.load_from_checkpoint("./models/model.ckpt", strict=False).eval().cuda()
model.hparams.threshold = -3.2
model.hparams.test_with_beam_search = False
model.hparams.test_with_beam_search_no_candidates = False

# loading the KB with the entities
model.generate_global_trie()

if args.full_candidate_sets:
    import pickle
    with open('candidate_list.pkl', 'rb') as f:
        # Load the list from the file
        full_candidates_list = pickle.load(f)
    full_candidates_list = [entity.replace("_", " ") for entity in full_candidates_list]

if __name__ == '__main__':
    annotator_name = "ea-el"

    try:
        app.run(host="localhost", port=int(os.environ.get("PORT", 3002)), debug=False)
    finally:
        if annotate_result:
            with open(f"annotate_{annotator_name}_result.json", "w", encoding="utf-8") as f:
                f.write(json.dumps({"annotations": annotate_result}))
