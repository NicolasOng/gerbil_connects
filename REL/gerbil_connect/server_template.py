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

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner
from threading import Thread
from queue import Queue

import argparse

wiki_version = "wiki_2014"
base_url = "/mnt/d/Datasets/Radboud/"
annotation_queue = Queue()

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

def REL_model(raw_text):
    output_queue = Queue()
    annotation_queue.put((raw_text, output_queue))
    output = output_queue.get()
    return output

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
        for annotation in REL_model(raw_text):
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


def REL_thread():
    # set up the MD and ED models
    mention_detection = MentionDetection(base_url, wiki_version)

    tagger_ner = load_flair_ner("ner-fast")
    #tagger_ngram = Cmns(base_url, wiki_version, n=5)

    if wiki_version == "wiki_2014":
        model_path = "ed-wiki-2014"
    elif wiki_version == "wiki_2019":
        model_path = "ed-wiki-2019"
    config = {
        "mode": "eval",
        "model_path": model_path,
    }
    Radboud_model = EntityDisambiguation(base_url, wiki_version, config)

    print("Model ready.")

    while True:
        raw_text, output_queue = annotation_queue.get()
        if raw_text is None:  # sentinel to stop the thread
            break

        # process the raw text into the input format
        doc_name = "my_doc"
        input_text = {
            doc_name: (raw_text, []),
        }
        # Execute the task and put the result in result_queue
        # mention detection using flair
        mentions, n_mentions = mention_detection.find_mentions(input_text, tagger_ner)

        if args.no_candidate_sets:
            # empties the candidate sets
            for mention in mentions[doc_name]:
                mention["candidates"] = []

        # entity disambiguation
        predictions, timing = Radboud_model.predict(mentions)
        
        result = process_results(mentions, predictions, input_text)

        # process results
        '''
        Example result format:
        {'my_doc': [(0, 13, 'Hello, world!', 'Hello_world_program', 0.6534378618767961, 182, '#NGRAM#')]}
        '''
        mention_list = result.get(doc_name, [])
        output = [(mention[0], mention[0] + mention[1], mention[3]) for mention in mention_list]
        output_queue.put(output)

# start

parser = argparse.ArgumentParser()
parser.add_argument("--no-candidate-sets", action="store_true")
args = parser.parse_args()

if __name__ == '__main__':
    annotator_name = "REL"

    Thread(target=REL_thread, daemon=True).start()

    try:
        app.run(host="localhost", port=int(os.environ.get("PORT", 3002)), debug=False)
    finally:
        if annotate_result:
            with open(f"annotate_{annotator_name}_result.json", "w", encoding="utf-8") as f:
                f.write(json.dumps({"annotations": annotate_result}))
