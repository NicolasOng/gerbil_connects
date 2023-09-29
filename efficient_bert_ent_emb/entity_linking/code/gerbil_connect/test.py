import os
import json
import pathlib
from threading import Lock
from flask import Flask, request
from flask_cors import CORS, cross_origin
from gerbil_connect.nif_parser import NIFParser

import argparse
import pickle

from run_aida import EntityLinkingAsLM

from gerbil_connect.helper_functions import aida_tokenize, token_to_character_index, split_by_whitespace, punkt_tokenize

from utils.util_wikidata import load_wikidata

import torch.nn as nn

def ebert_split_too_long_aida_tokens(sentence, model):
    '''
    This is how the EBERT evaluator splits up documents that are too long for the model - basically searches for the most central period and splits there.
    eee-bert uses the same code in this area of their run_aida.py file.
    '''
    if len(model.tokenizer.tokenize(" ".join(sentence + model.left_pattern + model.right_pattern))) >= model.max_len-2:
        midpoint = len(sentence) // 2
        breaking = False
        for i in range(0, midpoint - 5):
            if breaking: break
            for direction in (-1, 1):
                point = midpoint + (direction * i)
                if sentence[point] == ".":
                    midpoint = point+1
                    breaking = True
                    break

        sentence_a, sentence_b = sentence[:midpoint], sentence[midpoint:]

        left_list = ebert_split_too_long_aida_tokens(sentence_a, model)
        right_list = ebert_split_too_long_aida_tokens(sentence_b, model)
        return left_list + right_list
    else:
        return [sentence]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type = str, required = True)
    parser.add_argument("--bert_name", type = str, default = "bert-base-uncased")
    parser.add_argument("--mapper_name", type = str, default = "None")
    parser.add_argument("--ebert_name", type = str, default = "wikipedia2vec-base-cased")
    
    parser.add_argument("--train_file", type = str, default = "../data/AIDA/aida_train.txt")
    parser.add_argument("--dev_file", type = str, default = "../data/AIDA/aida_dev.txt")
    parser.add_argument("--test_file", type = str, default = "../data/AIDA/aida_test.txt")

    parser.add_argument("--wikidata_entity_types_path", type = str, default = "../../resources/wikidata_entity_types.tsv")

    parser.add_argument("--use_type_emb", dest = "use_type_emb", action = "store_true", default = True)
    parser.add_argument("--type_emb_option", type = str, default = "mix",
                        help="whether use type emb, surface emb or mix of them")
    parser.add_argument("--nouse_type_emb", dest = "use_type_emb", action = "store_false")

    parser.add_argument("--use_ebert_emb", dest = "use_ebert_emb", action = "store_true", default = False)
    parser.add_argument("--nouse_ebert_emb", dest = "use_ebert_emb", action = "store_false")

    parser.add_argument("--use_pos_emb", dest = "use_pos_emb", action = "store_true", default = True)
    parser.add_argument("--nouse_pos_emb", dest = "use_pos_emb", action = "store_false")

    parser.add_argument("--do_reinit_lm", action = "store_true")
    parser.add_argument("--do_predict_all_epochs", action = "store_true")
    
    parser.add_argument("--do_use_priors", dest = "do_use_priors", action = "store_true", default = True)
    parser.add_argument("--nodo_use_priors", dest = "do_use_priors", action = "store_false")
    parser.add_argument("--do_prime_mask", dest = "do_prime_mask", action = "store_true", default = False)
    parser.add_argument("--nodo_prime_mask", dest = "do_prime_mask", action = "store_false")

    parser.add_argument("--max_len", type = int, default = 512)
    parser.add_argument("--decode_iter", type = int, default = 1)
    parser.add_argument("--max_candidates", type = int, default = 1000)
    parser.add_argument("--epochs", type = int, default = 10)
    parser.add_argument("--warmup_proportion", type = float, default = 0.1)
    parser.add_argument("--null_penalty", type = float, default = 1.0)
    parser.add_argument("--device", type = int, default = 0)
    parser.add_argument("--lr", type = float, default = 2e-5)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--eval_batch_size", type = int, default = 4)
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 16)
    parser.add_argument("--granularity", type = str, default = "document", choices = ("document", "paragraph"))

    return parser.parse_args()

def load_model(args):
    with open(os.path.join(args.model_dir, "model_args.json")) as handle:
        model_args = json.load(handle)
    
    print(model_args, flush = True)
    model = EntityLinkingAsLM(**model_args, device = args.device)
    model.load(args.model_dir)

    return model

def predict_sentence(model, sentence, args):
    '''
    the model's predict_sentence method expects sentence to be in a tokenized format. EG:
    ['CRICKET', '-', 'ENGLISH', 'COUNTY', 'CHAMPIONSHIP', 'SCORES', '.', ..., '4-43', ')', ',', 'Durham', '114', '(', 'S.', 'Watkin', '4-28', ')', ... ]
    also, they can't be full documents - need to be split up.
    '''
    return model.predict_sentence(sentence, args.eval_batch_size, args.decode_iter)

def eeebert_model(raw_text):
    mode = "punkt"

    # convert the text given into a list of sentences,
    # where each sentence is a list of strings representing words.
    if mode == "gold":
        # tokenize as aida would (looks up from the aida file)
        tokens = aida_tokenize(raw_text)
    elif mode == "whitespace":
        # split the document by whitespace
        tokens = split_by_whitespace(raw_text)
    elif mode == "punkt":
        # tokenize using nltk's punkt tokenizer
        tokens = punkt_tokenize(raw_text)
    
    # use ebert's method of splitting documents that are too long
    sentence_list = ebert_split_too_long_aida_tokens(tokens, model)

    # use the model to predict spans/entities in each sentence.
    # spans use word/token indices
    # offset them by the sentence's starting token index so the spans
    # refer to the entire input's tokens, not just the sentence's
    preds = []
    offset = 0
    for sentence in sentence_list:
        sentence_preds = predict_sentence(model, sentence, args)
        sentence_preds = [[pred[0], offset + pred[1], offset + pred[2]] for pred in sentence_preds]
        offset += len(sentence)
        preds += sentence_preds

    # for every span, convert it from token indices to character-level indices.
    # remember for GERBIL it's text[start:start + span_length],
    # not text[start] to text[end]
    final_preds = []
    for pred in preds:
        entity_name = pred[0]
        span_start = pred[1]
        span_end = pred[2]
        s, e = token_to_character_index(tokens, raw_text, span_start, span_end)
        final_preds.append((s, e + 1, entity_name))

    return final_preds

args = parse_args()
print(os.uname(), flush = True)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", None), flush = True)
print(args, flush = True)

model = load_model(args)

use_type_emb =  args.use_type_emb
type_emb_option = args.type_emb_option
do_reinit_lm = args.do_reinit_lm
wikidata_entity_types_path = args.wikidata_entity_types_path

model.use_ebert_emb = args.use_ebert_emb
model.type_emb_option = args.type_emb_option
model.use_pos_emb = args.use_pos_emb
model.model_dir = args.model_dir
model.dev_file = args.dev_file
model.test_file = args.test_file

if use_type_emb and type_emb_option == 'mix':
    model.gate_hidden = nn.Linear(model.null_vector.shape[0],
                                    model.null_vector.shape[0])
    model.gate_hidden.to(device=model.device)

model.ent2idx = {None: 0}

if do_reinit_lm:
    for module in model.model.cls.predictions.transform.modules():
        model.model._init_weights(module)

if use_type_emb and not model.use_ebert_emb:
    model.wiki_title2qid, model.qid2label, model.qid2parent_qids = load_wikidata(wikidata_entity_types_path)

import pickle
from tqdm import tqdm

with open("gold_documents_new.pkl", 'rb') as file:
    new_gold_documents = pickle.load(file)

for doc in tqdm(new_gold_documents):
    eeebert_model(doc["raw_text"])
    break