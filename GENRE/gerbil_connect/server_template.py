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

from gerbil_connect.helper_functions import sentence_tokenize, sentence_to_document_character_index, aida_get_gold_document, character_to_character_index, punkt_tokenize, aida_tokenize

import sys
import pickle

import json

import urllib.parse

import argparse

with open('./gerbil_connect/config.json', 'r') as f:
    config = json.load(f)

MODEL_LOCATION = config['MODEL_LOCATION']
GENRE_REPO_LOCATION = config['GENRE_REPO_LOCATION']
FAIRSEQ_REPO_LOCATION = config['FAIRSEQ_REPO_LOCATION']
CANDIDATE_FILE = config['CANDIDATE_FILE']

sys.path.append(GENRE_REPO_LOCATION)
sys.path.append(FAIRSEQ_REPO_LOCATION)

from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_fairseq as get_entity_spans

from model import Model
from transform_predictions import compute_labels, get_mapping
from qid_to_wikipedia import qid_to_name

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

def genre_model1(raw_text):
    '''
    This function takes in raw text, runs it through the genre model,
    and returns the predictions.
    Splits into sentences as GENRE fails on longer strings,
    And gives GENRE a mention_trie and mention_to_candidate_dict to help
    it predict.
    These two objects provided by Elevant.
    '''
    # Split the raw text into sentences.
    split_into_sentences = False
    if split_into_sentences:
        sentences = sentence_tokenize(raw_text)
    else:
        sentences = [raw_text]
    
    # use the model to get the spans/entities
    # in the form [[(start, length, entity), ...], ...],
    # a list of spans for each given sentence.
    model_preds = get_entity_spans(model, sentences, mention_to_candidates_dict=mention_to_candidates_dict, mention_trie=mention_trie)

    # convert these from sentence to document spans
    # and to the correct format.
    final_preds = []
    if split_into_sentences:
        for sentence_i, sentence_preds in enumerate(model_preds):
            sentence_length = len(sentences[sentence_i])
            for pred in sentence_preds:
                s_start = pred[0]
                length = pred[1]
                entity = pred[2]
                if (s_start >= sentence_length or s_start + length > sentence_length):
                    # for some reason, GENRE sometimes predicts spans that start after the end of the sentence.
                    continue
                d_start = sentence_to_document_character_index(sentences, raw_text, sentence_i, s_start)
                final_pred = (d_start, d_start + length, entity)
                final_preds.append(final_pred)
    else:
        sentence_length = len(sentences[0])
        for pred in model_preds[0]:
            s_start = pred[0]
            length = pred[1]
            entity = pred[2]
            if (s_start >= sentence_length or s_start + length > sentence_length):
                continue
            final_pred = (s_start, s_start + length, entity)
            final_preds.append(final_pred)

    return final_preds

def genre_model2(raw_text):
    '''
    This one just reads Elevant's predictions for the given raw text,
    and returns the results.
    The model is run on all the text beforehand.
    '''
    # get the text with spaces
    aida_document = aida_get_gold_document(raw_text)
    text_with_spaces = aida_document["doc_spaces"]

    # find that in the list of GENRE predictions
    with open("dev-out.wiki_ids.jsonl", 'r') as dev_file, open("test-out.wiki_ids.jsonl", 'r') as test_file:
        genre_pred_line = None
        for line in dev_file:
            if genre_pred_line is not None: break
            line_data = json.loads(line)
            if (line_data["text"] == text_with_spaces):
                genre_pred_line = line_data
        for line in test_file:
            if genre_pred_line is not None: break
            line_data = json.loads(line)
            if (line_data["text"] == text_with_spaces):
                genre_pred_line = line_data
    
    # with the predictions, format them.
    final_preds = []
    for preds in genre_pred_line["entity_mentions"]:
        span = preds["span"]
        entity_id = preds['wiki_id']
        start = span[0]
        end = span[1]
        # need to fix the span indices because GENRE is fed a tokenized input here
        r_start = character_to_character_index(text_with_spaces, raw_text, start)
        r_end = character_to_character_index(text_with_spaces, raw_text, end - 1) + 1
        # this does stuff like: Zweibr%C3%BCcken -> Zweibrücken
        # not sure if this is what GERBIL wants...
        d_entity = urllib.parse.unquote(entity_id)
        pred = (r_start, r_end, d_entity)
        final_preds.append(pred)

    return final_preds

def genre_model3(raw_text):
    '''
    This function runs Elevant's model in real time,
    instead of just reading the predictions.
    Based off of their main.py and transform_predictions.py
    '''
    # tokenize the input text
    # Elevant's model ran on tokenized text, instead of the text like the one given.
    # eg: Lebed , Chechens sign framework political deal . KHASAVYURT , Russia 1996-08-31
    tokenize_mode = "punkt"
    if tokenize_mode == "punkt":
        tokenized_string = " ".join(punkt_tokenize(raw_text))
    elif tokenize_mode == "aida":
        tokenized_string = " ".join(aida_tokenize(raw_text))
    
    # prepare the text
    evaluation_span = (0, len(tokenized_string))
    before = tokenized_string[:evaluation_span[0]]
    after = tokenized_string[evaluation_span[1]:]
    text = tokenized_string[evaluation_span[0]:evaluation_span[1]]

    # run the model
    prediction = model.predict_iteratively(text)
    genre_text = before + prediction + after

    # convert from genre text to (start, end, label)
    wikipedia_labels = compute_labels(tokenized_string, genre_text, 0)

    # convert from label to qid to wiki name, and fix the span indices.
    final_preds = []
    for start, end, label in wikipedia_labels:
        qid = label
        if wikipedia:
            qid = "https://en.wikipedia.org/wiki/" + label.replace(" ", "_")
        else:
            if label in mapping:
                qid = mapping[label]
            elif label in redirects:
                redirected = redirects[label]
                if redirected in mapping:
                    qid = mapping[redirected]
        print(start, end, label, qid)
        wiki_name = qid_to_name.get(qid, "")
        # need to fix the span indices because GENRE is fed a tokenized input here
        r_start = character_to_character_index(tokenized_string, raw_text, start)
        r_end = character_to_character_index(tokenized_string, raw_text, end - 1) + 1
        # this does stuff like: Zweibr%C3%BCcken -> Zweibrücken
        # not sure if this is what GERBIL wants...
        d_entity = urllib.parse.unquote(wiki_name)
        pred = (r_start, r_end, d_entity)
        final_preds.append(pred)
    
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
        for annotation in genre_model3(raw_text):
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

if __name__ == '__main__':
    annotator_name = "GENRE"
    genre_mode = "genre3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-candidate-sets", action="store_true")
    args = parser.parse_args()

    if (genre_mode == "genre1"):
        model = GENRE.from_pretrained(MODEL_LOCATION).eval()

        with open("mention_to_candidates_dict.pkl", "rb") as f:
            mention_to_candidates_dict = pickle.load(f)
        
        with open("mention_trie.pkl", "rb") as f:
            mention_trie = pickle.load(f)
    elif genre_mode == "genre3":
        print("load model...")
        if args.use_candidate_sets:
            print("...with candidate sets...")
            model = Model(yago=True,
                        mention_trie="data/mention_trie.pkl",
                        mention_to_candidates_dict="data/mention_to_candidates_dict.pkl",
                        candidates_trie=None)
        else:
            print("...without candidate sets...")
            model = Model(yago=True,
                        mention_trie=None,
                        mention_to_candidates_dict=None,
                        candidates_trie=None)
        wikipedia = False
        if not wikipedia:
            print("read mapping...")
            mapping = get_mapping()

            print("load redirects...")
            with open("data/elevant/link_redirects.pkl", "rb") as f:
                redirects = pickle.load(f)

    try:
        app.run(host="localhost", port=int(os.environ.get("PORT", 3002)), debug=False)
    finally:
        if annotate_result:
            with open(f"annotate_{annotator_name}_result.json", "w", encoding="utf-8") as f:
                f.write(json.dumps({"annotations": annotate_result}))
