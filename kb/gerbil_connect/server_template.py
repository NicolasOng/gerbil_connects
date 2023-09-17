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

from kb.include_all import TokenizerAndCandidateGenerator

from allennlp.data import DatasetReader, Vocabulary, DataIterator
from allennlp.common import Params

from allennlp.nn.util import move_to_device

from allennlp.models.archival import load_archive

from allennlp.data import Instance
from allennlp.data.dataset import Batch

import torch
import re

reader_params_ww = Params({
        "type": "aida_wiki_linking",
        "entity_disambiguation_only": False,
        "entity_indexer": {
            "type": "characters_tokenizer",
            "namespace": "entity_wiki",
            "tokenizer": {
                "type": "word",
                "word_splitter": {
                    "type": "just_spaces"
                }
            }
        },
        "extra_candidate_generators": {
            "wordnet": {
                "type": "wordnet_mention_generator",
                "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl"
            }
        },
        "should_remap_span_indices": True,
        "token_indexers": {
            "tokens": {
                "type": "bert-pretrained",
                "do_lowercase": True,
                "max_pieces": 512,
                "pretrained_model": "bert-base-uncased",
                "use_starting_offsets": True,
            }
        }
    })

reader_params_w = Params({
    "type": "aida_wiki_linking",
    "entity_disambiguation_only": False,
    "token_indexers": {
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased",
            "do_lowercase": True,
            "use_starting_offsets": True,
            "max_pieces": 512,
        },
    },
    "entity_indexer": {
        "type": "characters_tokenizer",
        "tokenizer": {
            "type": "word",
            "word_splitter": {"type": "just_spaces"},
        },
        "namespace": "entity",
    },
    "should_remap_span_indices": True,
})

cg_params_ww = Params({
                "type": "bert_tokenizer_and_candidate_generator",
                "bert_model_type": "bert-base-uncased",
                "do_lower_case": True,
                "entity_candidate_generators": {
                    "wordnet": {
                        "type": "wordnet_mention_generator",
                        "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl"
                    }
                },
                "entity_indexers": {
                    "wordnet": {
                        "type": "characters_tokenizer",
                        "namespace": "entity_wordnet",
                        "tokenizer": {
                            "type": "word",
                            "word_splitter": {
                                "type": "just_spaces"
                            }
                        }
                    }
                }
            })

def split_into_sentences(text):
    # Split on periods, question marks, or exclamation points that are followed by whitespace and an uppercase letter.
    # Exclude cases where the period is part of an abbreviation or initial.
    pattern = r'(?<=[.!?])\s+(?=[A-Z])(?<!\b[A-Z]\.\s)(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z][a-z][a-z]\.)'
    
    # Find the start indices of all matches
    split_indices = [0] + [match.end() for match in re.finditer(pattern, text)]
    
    # Split the text at the found indices
    sentences = [text[i:j].strip() for i, j in zip(split_indices, split_indices[1:] + [None])]
    
    # Return sentences along with their starting indices
    return [(sentence, text.index(sentence)) for sentence in sentences if sentence]

def map_entity_id_to_name(target_id, entity_ids, entity_names):
    """
    Retrieve the name corresponding to the given ID.

    :param entity_names: 2D list containing names of entities.
    :param entity_ids: 2D list containing IDs of entities.
    :param target_id: The ID for which we want the corresponding name.
    :return: Corresponding name or None if not found.
    """
    ids_list = entity_ids[0].tolist()

    for i, row in enumerate(ids_list):
        for j, entity_id in enumerate(row):
            if entity_id == target_id:
                return entity_names[i][j]
    return None

def get_word_span(token_span, token_spans, word_spans):
    """
    Given a token span, return the corresponding word span.

    :param token_span: The token span for which we want the corresponding word span.
    :param token_spans: Tensor containing token spans.
    :param word_spans: List of word spans.
    :return: Corresponding word span.
    """
    # Convert token_span to a tensor and ensure it's on the CPU
    token_span_tensor = torch.tensor(token_span).cpu()
    
    # Ensure token_spans is also on the CPU
    token_spans_cpu = token_spans.cpu()
    
    # Find the index of the token_span in token_spans
    matches = (token_spans_cpu[0] == token_span_tensor).all(dim=1)
    index = torch.nonzero(matches, as_tuple=True)[0]
    
    # If the token span is not found, return None
    if len(index) == 0:
        return None
    
    # Return the corresponding word span
    return word_spans[index[0].item()]

def word_to_char_span(word_span, words):
    '''
    given a word span like (0, 1), referring to a span including the words "The Apple" in the words "The Apple is a great fruit", it is converted to a span like (0, 9).
    '''
    # Compute the start position of each word
    word_starts = [0]
    for word in words[:-1]:
        word_starts.append(word_starts[-1] + len(word) + 1)  # +1 for the space
    
    start = word_span[0]
    end = word_span[1]
    char_start = word_starts[start]
    char_end = word_starts[end] + len(words[end])  # add -1 if end index is inclusive

    return (char_start, char_end)

def convert_decoded_output(decode_output, entity_ids, entity_names, token_spans, word_spans, words):
    '''
    input: [(0, (1, 2), 320843), (0, (7, 8), 426103), ... ]
    output: [(65, 71, 'London'), (83, 94, 'West_Indies_cricket_team'), ...]
    '''
    converted_output = []
    for _, token_span, entity_id in decode_output:
        entity_name = map_entity_id_to_name(entity_id, entity_ids, entity_names)
        word_span = get_word_span(token_span, token_spans, word_spans)
        character_span = word_to_char_span(word_span, words)
        mention = (character_span[0], character_span[1], entity_name)
        converted_output.append(mention)
    return converted_output

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

def knowbert_model(raw_text):
    sentences = split_into_sentences(raw_text)

    instances = []
    processeds = []
    for sentence, _ in sentences:
        processed = reader.mention_generator.get_mentions_raw_text(sentence, whitespace_tokenize=True)
        # get_mentions_raw_text returns a very slightly different dict from get_mentions_with_gold
        processed["candidate_entity_prior"] = processed["candidate_entity_priors"]
        del processed["candidate_entity_priors"]
        instance = reader.text_to_instance(doc_id="", **processed)
        processeds.append(processed)
        instances.append(instance)

    total_final_output = []
    for batch_no, batch in enumerate(iterator(instances, shuffle=False, num_epochs=1)):
        b = move_to_device(batch, 0)

        b['candidates'] = {'wiki': {
                'candidate_entities': b.pop('candidate_entities'),
                'candidate_entity_priors': b.pop('candidate_entity_prior'),
                'candidate_segment_ids': b.pop('candidate_segment_ids'),
                'candidate_spans': b.pop('candidate_spans')}}

        if is_wordnet_and_wiki:
            extra_candidates = b.pop('extra_candidates')
            seq_len = b['tokens']['tokens'].shape[1]
            bbb = []
            for e in extra_candidates:
                for k in e.keys():
                    e[k]['candidate_segment_ids'] = [0] * len(e[k]['candidate_spans'])
                ee = {'tokens': ['[CLS]'] * seq_len, 'segment_ids': [0] * seq_len,
                        'candidates': e}
                ee_fields = candidate_generator.convert_tokens_candidates_to_fields(ee)
                bbb.append(Instance(ee_fields))
            eb = Batch(bbb)
            eb.index_instances(vocab)
            padding_lengths = eb.get_padding_lengths()
            tensor_dict = eb.as_tensor_dict(padding_lengths)
            b['candidates'].update(tensor_dict['candidates'])
            bb = move_to_device(b, 0)
        else:
            bb = b

        raw_output = model(**bb)
    
        decoded_output = wiki_el._decode(raw_output["wiki"]["linking_scores"], bb["candidates"]["wiki"]["candidate_spans"], bb["candidates"]["wiki"]["candidate_entities"]["ids"])
    
        final_output = convert_decoded_output(decoded_output, bb["candidates"]["wiki"]["candidate_entities"]["ids"], processeds[batch_no]["candidate_entities"], bb["candidates"]["wiki"]["candidate_spans"], processeds[batch_no]["candidate_spans"], processeds[batch_no]["tokenized_text"])

        sentence_start_index = sentences[batch_no][1]
        final_output = [(a + sentence_start_index, b + sentence_start_index, c) for a, b, c in final_output]

        total_final_output += final_output

    
    return total_final_output

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
        for annotation in knowbert_model(raw_text):
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
    annotator_name = "KnowBert"
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--model_archive', type=str)
    parser.add_argument('--wiki_and_wordnet', action='store_true')

    args = parser.parse_args()

    model_archive_file = args.model_archive
    is_wordnet_and_wiki = args.wiki_and_wordnet

    archive = load_archive(model_archive_file)
    params = archive.config
    vocab = Vocabulary.from_params(params.pop('vocabulary'))

    model = archive.model
    model.cuda()
    model.eval()

    wiki_el = getattr(model, "wiki_soldered_kg").entity_linker
        
    if is_wordnet_and_wiki:
        reader_params = reader_params_ww
    else:
        reader_params = reader_params_w
    
    if is_wordnet_and_wiki:
        candidate_generator = TokenizerAndCandidateGenerator.from_params(cg_params_ww)
    
    reader = DatasetReader.from_params(Params(reader_params))

    iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 1}))
    iterator.index_with(vocab)

    try:
        app.run(host="localhost", port=int(os.environ.get("PORT", 3002)), debug=False)
    finally:
        if annotate_result:
            with open(f"annotate_{annotator_name}_result.json", "w", encoding="utf-8") as f:
                f.write(json.dumps({"annotations": annotate_result}))
