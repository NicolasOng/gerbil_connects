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

import argparse
from gerbil_experiments.nn_processing import Annotator

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

def print_results(raw_text, output):
    for start, end, entity in output:
        print(start, end, entity)
        print(raw_text[start:end])
        print(raw_text[start-10:start] + "*" + raw_text[start:end] + "*" + raw_text[end:end+10])
        print("")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default='logs/log.txt', type=str,
                        help='log path')
    parser.add_argument('--blink_dir', default='blink_model/', type=str,
                        help='blink pretrained bi-encoder path')
    parser.add_argument(
        "--passage_len", type=int, default=32,
        help="the length of each passage"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="length of stride when chunking passages",
    )
    parser.add_argument('--bsz_retriever', type=int, default=8192,
                        help='the batch size of retriever')
    parser.add_argument('--max_len_retriever', type=int, default=42,
                        help='max length of the retriever input passage ')
    parser.add_argument('--retriever_path', default='model_retriever/retriever.pt', type=str,
                        help='trained retriever path')
    parser.add_argument('--type_retriever_loss', type=str,
                        default='sum_log_nce',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='type of marginalize for retriever')
    parser.add_argument('--gpus', default='0,1', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--cands_embeds_path', default='candidates_embeds/candidate_embeds.npy', type=str,
                        help='the path of candidates embeddings')
    parser.add_argument('--k', type=int, default=100,
                        help='top-k candidates for retriever')
    parser.add_argument('--ents_path', default='kb/entities_kilt.json', type=str,
                        help='entity file path')
    parser.add_argument('--max_len_reader', type=int, default=180,
                        help='max length of joint input [%(default)d]')
    parser.add_argument('--max_num_candidates', type=int, default=100,
                        help='max number of candidates [%(default)d] when '
                             'eval for reader')
    parser.add_argument('--bsz_reader', type=int, default=32,
                        help='batch size [%(default)d]')
    parser.add_argument('--reader_path', default='model_reader/reader.pt', type=str,
                        help='trained reader path')
    parser.add_argument('--type_encoder', type=str,
                        default='squad2_electra_large',
                        help='the type of encoder')
    parser.add_argument('--type_span_loss', type=str,
                        default='sum_log',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='the type of marginalization for reader')
    parser.add_argument('--type_rank_loss', type=str,
                        default='sum_log',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='the type of marginalization for reader')
    parser.add_argument('--num_spans', type=int, default=3,
                        help='top num_spans for evaluation on reader')
    parser.add_argument('--thresd', type=float, default=0.05, #1.5e-5
                        help='probabilty threshold for evaluation on reader')
    parser.add_argument('--max_answer_len', type=int, default=10,
                        help='max length of answer [%(default)d]')
    parser.add_argument('--max_passage_len', type=int, default=32,
                        help='max length of question [%(default)d] for reader')
    parser.add_argument('--document', type=str,
                        help='test document')
    parser.add_argument('--save_span_path', type=str,
                        help='save span-based document-level results path')
    parser.add_argument('--save_char_path', type=str,
                        help='save char-based path')
    parser.add_argument('--add_topic', action='store_true',
                        help='add title?')
    parser.add_argument('--do_rerank', action='store_true',
                        help='do reranking for reader?')
    parser.add_argument('--use_title', action='store_true',
                        help='use title?')
    parser.add_argument('--no_multi_ents', action='store_true',
                        help='no repeated entities are allowed given a span?')
    
    parser.add_argument('--candidate_set_setting', type=str)

    args = parser.parse_args()
    return args

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

def entqa_model(raw_text):
    # nnprocessing.process expects a second parameter, given_spans
    response = entqa_annotator.get_predicts(raw_text)
    # the response is in (start, length, name) instead of (start, end, name).
    converted_response = [(start, start + length, name) for start, length, name in response]

    print_results(raw_text, converted_response)

    return converted_response

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
        for annotation in entqa_model(raw_text):
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

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
entqa_annotator = Annotator(args)

if args.candidate_set_setting == "empty1":
    entqa_annotator.set_empty1_setting()

if args.candidate_set_setting == "empty2":
    entqa_annotator.set_empty2_setting()

if args.candidate_set_setting == "full1":
    entqa_annotator.set_full1_setting()

if args.candidate_set_setting == "full2":
    entqa_annotator.set_full2_setting()

if __name__ == '__main__':
    annotator_name = "EntQA"
    try:
        app.run(host="localhost", port=int(os.environ.get("PORT", 3002)), debug=False)
    finally:
        if annotate_result:
            with open(f"annotate_{annotator_name}_result.json", "w", encoding="utf-8") as f:
                f.write(json.dumps({"annotations": annotate_result}))
