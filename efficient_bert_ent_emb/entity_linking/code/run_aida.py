import sys
import os
import pickle

import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)

from pytorch_transformers import *
from emb_input_transformers import *
from embeddings import MappedEmbedding, load_embedding
#  from mappers import load_mapper

from kb.wiki_linking_util import WikiCandidateMentionGenerator
from utils.util_wikidata import load_wikidata
from tqdm import tqdm, trange
from torch.autograd import Variable
import re
import argparse
import json
import copy

def normalize_entity(ebert_emb, entity, prefix = ""):
    entity_no_prefix = entity
    entity = prefix + entity

    if entity is None:
        return None

    if not entity in ebert_emb:
        return entity_no_prefix
        #  return None

    true_title = ebert_emb.index(entity).title
    true_entity = "_".join(ebert_emb.index(entity).title.split())

    assert prefix + true_entity in ebert_emb
    return true_entity


def extract_spans(in_file, out_file, kb = None):
    """
    Extract gold spans from AIDA file into a more handy format.

    in_file : File in standard AIDA format
    out_file : TSV with format start -- end -- entity (wiki) id -- surface form

    (Note that start and end positions are calculated w.r.t. to the entire file.)
    """
    if kb is not None:
        kb = load_embedding(kb)

    counter = 0
    flag = False
    spans = []
    with open(in_file) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("DOCSTART") or line.startswith("DOCEND") or line == "*NL*" or len(line) == 0:
                continue
            elif line.startswith("MMSTART"):
                assert not flag
                entity = line.strip().split()[-1]

                if kb is not None:
                    entity = normalize_entity(kb, entity, kb.prefix)

                spans.append([entity, [], counter, counter-1])
                flag = True
            elif line.startswith("MMEND"):
                flag = False
            elif flag:
                spans[-1][-1] += 1
                spans[-1][1].append(line)
                counter += 1
            else:
                counter += 1

    spans.sort(key = lambda x:(x[-2], x[-1]))

    with open(out_file, "w") as whandle:
        for entity, surface, start, end in spans:
            surface = " ".join(surface)
            whandle.write(f"{start}\t{end}\t{entity}\t{surface}\n")


class Sample:
    def __init__(self, input_ids, tokenized, mask_pos, correct_idx, candidate_ids, biases = None, candidates = None, start = None, end = None, sentence = None, data_idx = None):
        self.input_ids = input_ids
        self.mask_pos = mask_pos
        self.correct_idx = correct_idx
        self.candidate_ids = candidate_ids
        self.biases = biases

        self.tokenized = tokenized
        self.candidates = candidates
        self.start = start
        self.end = end
        self.data_idx = data_idx
        self.sentence = sentence

    def __str__(self):
        string = "*" * 50 + "\n"
        if self.data_idx:
            string += "Data Idx: ({})\n\n".format(",".join(["{}".format(i) for i in self.data_idx]))
        if self.sentence:
            string += "Orig Sentence: {}\n\n".format(" ".join(["{}".format(i) for i in self.sentence]))
        if self.start and self.end:
            string += "Start, end: ({},{})\n\n".format(self.start, self.end)
        string += "InputIDs: {}\n\n".format(" ".join(["{}".format(i) if j != self.mask_pos else "->{}<-".format(i) for j,i in enumerate(self.input_ids)]))
        string += "Tokens: {}\n\n".format(" ".join(["{}".format(i) if j != self.mask_pos else "->{}<-".format(i) for j,i in enumerate(self.tokenized)]))
        string += "CandidateIDs: {}\n\n".format(" ".join(["{}".format(i) if j != self.correct_idx else "->{}<-".format(i) for j,i in enumerate(self.candidate_ids)]))
        if self.candidates:
            string += "Candidates: {}\n\n".format(" ".join(["{}".format(i) if j != self.correct_idx else "->{}<-".format(i) for j,i in enumerate(self.candidates)]))
        if self.biases:
            string += "Biases: {}\n\n".format(" ".join(["{:.3}".format(i) if not i is None else "NIL" for i in self.biases]))
        string += "*" * 50
        return string

    def __repr__(self):
        return self.__str__()

class EntityLinkingAsLM:
    NO_DECAY = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    SPECIAL_ENT_RGX = re.compile("@@.+?@@")

    def __init__(self, 
            bert_name = "bert-base-cased", 
            ebert_name = "wikipedia2vec-base-cased",
            mapper_name = "linear",
            device = 0, 
            ent_prefix = "ENTITY/",
            left_pattern = ["[MASK]", "/"],
            right_pattern = ["*"],
            granularity = "document",
            max_len = 512,
            margin = 1,
            dynamic_max_len = True,
            max_candidates = 1000,
            do_use_priors = False,
            do_prime_mask = False,
            seed = 0):

        """
        bert_name : name of bert model (compatible with pytorch_transformers)
        ebert_name : name of ebert embeddings (file $ebert_name$ should exist)
        device : CUDA device
        ent_prefix : prefix used to distinguish entities from words
        left_pattern : pattern introduced before candidate mention
        right_pattern : pattern introduced after candidate mention
        (one of left_pattern, right_pattern should contain [MASK])
        granularity : splitting AIDA at document or paragraph level
        max_len : maximum length of input to BERT
        margin : margin for max margin training (not used)
        dymamic_max_len : whether to use dynamic padding
        max_candidates : number of candidates per potential mention
        do_use_priors : whether to use candidate generator priors (not recommended)
        do_prime_mask : whether to prime [MASK] token with average candidate (recommended)
        seed : random seed
        """

        self.dynamic_max_len = dynamic_max_len
        self.device = device
        self.max_len = max_len
        self.left_pattern = left_pattern
        self.right_pattern = right_pattern
        self.granularity = granularity
        self.ent_prefix = ent_prefix

        self.rnd = np.random.RandomState(seed)
        self.do_use_priors = do_use_priors
        self.do_prime_mask = do_prime_mask
        
        assert self.granularity in ("paragraph", "document")
        assert self.max_len <= 512
        assert len([x for x in self.left_pattern + self.right_pattern if x == "[MASK]"]) == 1
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case = "uncased" in bert_name)
        
        assert not any([w.startswith(ent_prefix) for w in self.tokenizer.vocab.keys()])
        
        self.ebert_emb = load_embedding(ebert_name, prefix = self.ent_prefix)
        if mapper_name and (mapper_name != "None"):
            self.ebert_emb = MappedEmbedding(self.ebert_emb, load_mapper(f"{ebert_name}.{bert_name}.{mapper_name}.npy"))

        tmp_bert_model = BertModel.from_pretrained(bert_name)
        self.bert_pos_emb = tmp_bert_model.embeddings.to(device=self.device)
        self.bert_emb = self.bert_pos_emb.word_embeddings
        #  self.bert_emb = tmp_bert_model.embeddings.word_embeddings
        del tmp_bert_model
         
        self.model = EmbInputBertForMaskedEmbLM.from_pretrained(bert_name).to(device = self.device)
        
        null_vector = self.rnd.uniform(low = -self.model.config.initializer_range, high = self.model.config.initializer_range, size = (self.model.config.hidden_size,))
        self.null_vector = Variable(torch.tensor(null_vector).to(dtype = torch.float, device = self.device), requires_grad = True)
        self.candidate_generator = WikiCandidateMentionGenerator(entity_world_path = None, max_candidates = max_candidates) 
        
        if self.do_use_priors:
            self.null_bias = Variable(torch.zeros((1,)).to(dtype = torch.float, device = self.device), requires_grad = True)
         

    def score_f1(self, true, pred):
        assert len(pred) == len(true)
        guessed, gold, correct = 0, 0, 0

        for t, p in zip(true, pred):
            if t == 0 and p == 0: pass
            elif t== 0 and p != 0: guessed += 1
            elif t != 0 and p == 0: gold += 1
            else:
                gold += 1
                guessed += 1
                if t == p: correct += 1
   
        prec_micro = 1.0 if guessed == 0 else correct / guessed
        rec_micro = 0.0 if gold == 0 else correct / gold
        f1_micro = 0 if prec_micro + rec_micro == 0 else 2 * prec_micro * rec_micro / (prec_micro + rec_micro)

        return prec_micro, rec_micro, f1_micro

    def printing_samples(self, data, out_path, verbose = True):

        def get_types(wiki_title):
            ent_types = []
            if (wiki_title in self.wiki_title2qid and
                self.wiki_title2qid[wiki_title] in self.qid2parent_qids):
                qid = self.wiki_title2qid[wiki_title]
                parent_qids = self.qid2parent_qids[qid]
                for qid in parent_qids:
                    if qid in self.qid2label:
                        label = self.qid2label[qid]
                        ent_types.append(label)
            return ent_types

        fh = open(out_path, "w")
        samples = []
        for data_idx, (sentence, spans) in tqdm(data.items(), disable = not verbose, desc = "Converting data to samples"):
            mentions = self.candidate_generator.get_mentions_raw_text(" ".join(sentence), whitespace_tokenize=True)
            span2candidates = {}

            span2gold = {(start, end): normalize_entity(self.ebert_emb, entity, self.ent_prefix) for entity, start, end in spans}
            prev_start, prev_end = None, None
            for (start, end), entities, priors in zip(mentions["candidate_spans"], mentions["candidate_entities"], mentions["candidate_entity_priors"]):
                if any([x.startswith(self.ent_prefix) or x == "[PAD]" for x in sentence[start:end+1]]):
                    continue

                normalized = [normalize_entity(self.ebert_emb, entity, self.ent_prefix) for entity in entities]

                valid_entities = []
                valid_entities_set = set()
                valid_priors = []

                for entity, prior in zip(normalized, priors):
                    if entity is None or entity in valid_entities_set:
                        continue

                    valid_entities.append(entity)
                    valid_priors.append(prior)
                    valid_entities_set.add(entity)

                biases = [-1.0] + [float(f"{x:.3f}") for x in valid_priors]
                entities = [None] + valid_entities

                if prev_end is not None:
                    tokens = sentence[prev_end+1:start]
                    for token in tokens:
                        fh.write(token + "\n")
                if len(entities) > 1:
                    span2candidates[(start, end)] = (entities, biases)
                    tag = "No_Gold"
                    gold_entity = None
                    if (start, end) in span2gold:
                        gold_entity = span2gold[(start, end)]
                        gold_types = get_types(gold_entity)
                        tag = f"{gold_entity}-{gold_types}"
                    entities_types = [get_types(entity) for entity in entities]
                    cand_tag = []
                    sorted_zip = sorted(zip(entities_types, entities, biases),
                                    key=lambda x:x[2], reverse=True)
                    for entity_types, entity, biase in sorted_zip:
                        if gold_entity == entity:
                            cand_tag.append(f"{biase}--{entity}(True)-{entity_types}")
                        else:
                            cand_tag.append(f"{biase}--{entity}-{entity_types}")
                    cand_tag = "  ".join(cand_tag)
                    tag = f"{tag} | {cand_tag}"
                    fh.write(f"{sentence[start:end+1]} : {tag}\n")
                else:
                    fh.write(f"{sentence[start:end+1]}\n")
                prev_start, prev_end = start, end
            fh.write("\n\n")
        fh.close()

    def data2samples(self, data, verbose = True):
        samples = []
        for data_idx, (sentence, spans) in tqdm(data.items(), disable = not verbose, desc = "Converting data to samples"):
            mentions = self.candidate_generator.get_mentions_raw_text(" ".join(sentence), whitespace_tokenize=True)
            span2candidates = {}

            for (start, end), entities, priors in zip(mentions["candidate_spans"], mentions["candidate_entities"], mentions["candidate_entity_priors"]):
                if any([x.startswith(self.ent_prefix) or x == "[PAD]" for x in sentence[start:end+1]]):
                    continue

                normalized = [normalize_entity(self.ebert_emb, entity, self.ent_prefix) for entity in entities]

                valid_entities = []
                valid_entities_set = set()
                valid_priors = []

                for entity, prior in zip(normalized, priors):
                    if entity is None or entity in valid_entities_set:
                        continue
                    
                    valid_entities.append(entity)
                    valid_priors.append(prior)
                    valid_entities_set.add(entity)

                biases = [None] + [np.log(x) for x in valid_priors]
                entities = [None] + valid_entities

                if len(entities) > 1:
                    span2candidates[(start, end)] = (entities, biases)
            
            span2gold = {(start, end): normalize_entity(self.ebert_emb, entity, self.ent_prefix) for entity, start, end in spans}
            
            for (start, end), (candidates, biases) in span2candidates.items():
                assert not ("[CLS]" in sentence or "[MASK]" in sentence or "[SEP]" in sentence or "[UNK]" in sentence)

                gold_candidate = span2gold.get((start, end), None)
                if not gold_candidate in candidates:
                    continue 
                
                correct_idx = candidates.index(gold_candidate)
               
                for entity in candidates:
                    if not entity in self.ent2idx:
                        self.ent2idx[entity] = len(self.ent2idx)
                
                sentence_with_pattern = sentence[:start] + self.left_pattern + sentence[start:end+1] + self.right_pattern + sentence[end+1:]

                present_entities = [token for token in sentence_with_pattern if token.startswith(self.ent_prefix)]
                sentence_with_pattern_ent = ["[UNK]" if token.startswith(self.ent_prefix) else token for token in sentence_with_pattern]
                sample_tokenized = self.tokenizer.tokenize(" ".join(sentence_with_pattern_ent))

                ent_pos = [i for i, token in enumerate(sample_tokenized) if token == "[UNK]"]
                del_pos = [i for i, token in enumerate(sample_tokenized) if token == "[PAD]"]

                assert len(ent_pos) == len(present_entities)
                for pos, ent in zip(ent_pos, present_entities):
                    sample_tokenized[pos] = ent

                for pos in sorted(del_pos, reverse = True):
                    del sample_tokenized[pos]

                if not self.do_use_priors:
                    biases = None

                mask_pos = sample_tokenized.index("[MASK]")

                sample_tokenized = ["[CLS]"] + sample_tokenized + ["[SEP]"]
                input_ids = self.tokenizer.convert_tokens_to_ids(sample_tokenized)

                samples.append(Sample(input_ids = input_ids,
                    tokenized = sample_tokenized,
                    mask_pos = sample_tokenized.index("[MASK]"),
                    correct_idx = correct_idx,
                    candidate_ids = [self.ent2idx[candidate] for candidate in candidates],
                    biases = biases,
                    sentence = sentence,
                    candidates = candidates,
                    start = start, end = end, data_idx = data_idx))

        if verbose:
            for i in self.rnd.randint(low = 0, high = len(samples), size = (5,)):
                print(samples[i], flush = True)

        return samples


    def read_aida_file(self, f, ignore_gold = True):

        data, sentence, spans = {}, [], []
        flag = False
        doc_count = -1
        
        with open(f) as handle:
            for line in handle:
                line = line.strip()
                if len(line) == 0: continue

                if line.startswith("DOCSTART"):
                    doc_count += 1
                    in_doc_count = -1
                if (line == "*NL*" and self.granularity == "paragraph") or line == "DOCEND":
                    if len(sentence):
                        in_doc_count += 1
                        if ignore_gold:
                            data[(doc_count, in_doc_count)] = [sentence, []]
                        else:
                            data[(doc_count, in_doc_count)] = [sentence, spans]
                    sentence, spans = [], []
                elif line == "*NL*":
                    pass
                elif line.startswith("DOCSTART"):
                    doc_count += 1
                elif line.startswith("MMEND"):
                    flag = False
                elif flag:
                    spans[-1][-1] += 1
                    sentence.append(line.split()[0])
                elif line.startswith("MMSTART"):
                    assert not flag
                    gold_entity = line.split()[-1]
                    spans.append([gold_entity, len(sentence), len(sentence)-1])
                    flag = True
                else: 
                    sentence.append(line.split()[0])

        if 1:
            delete, checked = set(), set()
        
            while len(delete) + len(checked) != len(data):
                for data_idx, (sentence, spans) in data.items():
                    if (not data_idx in delete) and (not data_idx in checked):
                        if len(self.tokenizer.tokenize(" ".join(sentence + self.left_pattern + self.right_pattern))) >= self.max_len-2:
                            midpoint = len(sentence) // 2
                            breaking = False
                            for i in range(0, midpoint - 5):
                                if breaking: break
                                for direction in (-1, 1):
                                    point = midpoint + (direction * i)
                                    if sentence[point] == "." and not any([x[1] <= point+1 and x[2] >= point+1 for x in spans]):
                                        midpoint = point+1
                                        breaking = True
                                        break

                            sentence_a, sentence_b = sentence[:midpoint], sentence[midpoint:]
                    
                            spans_a = [x for x in spans if x[2] < midpoint]
                            spans_b = [(x[0], x[1]-midpoint, x[2]-midpoint) for x in spans if x[2] > midpoint]
                        
                            data[data_idx + (1,)] = [sentence_a, spans_a]
                            data[data_idx + (2,)] = [sentence_b, spans_b]
                            delete.add(data_idx)
                            break

                        else:
                            checked.add(data_idx)
        
            assert len(delete.intersection(checked)) == 0
            for data_idx in delete:
                del data[data_idx]

        return data
       
    def normalize_predictions(self, data, predictions, sort_by_null = True):
        data_idx_offsets = {}
        current_offset = 0
        text = []
        for data_idx in sorted(list(data.keys())):
            data_idx_offsets[data_idx] = current_offset
            current_offset += len(data[data_idx][0])
            text.extend(data[data_idx][0])

        span2prediction = {}
        for entity, data_idx, start, end, ep in predictions:
            if entity is None: continue

            start += data_idx_offsets[data_idx]
            end += data_idx_offsets[data_idx]
            assert not (start, end) in span2prediction
            assert ep[0][0] is None
            null_p = ep[0][1]
            ep.sort(key = lambda x:x[1], reverse = True)
            assert ep[0][0] == entity
            span2prediction[(start, end)] = (entity, text[start:end+1], ep[0][1], null_p)
        
        if sort_by_null:
            spans_sorted = sorted(list(span2prediction.keys()), key = lambda x:span2prediction[x][3])
        else:
            spans_sorted = sorted(list(span2prediction.keys()), key = lambda x:span2prediction[x][2], reverse = True)

        
        blocked = set()
        for start, end in spans_sorted:
            for x in range(start, end+1):
                if x in blocked:
                    del span2prediction[(start, end)]
                    break

            if (start, end) in span2prediction:
                blocked.update(range(start, end+1))

        return span2prediction


    def _predict_sentence(self, sentence, batch_size = 4):#, gold_spans = None):
        #  self.ent2idx = {None: 0}
        
        samples = self.data2samples({(0,): [sentence, []]}, verbose = False)
        predictions = self.pred_loop(samples, batch_size = batch_size, verbose = False)
        span2prediction = {(start, end): ents_and_probas for _, _, start, end, ents_and_probas in predictions}
        assert all([span2prediction[key][0][0] is None for key in span2prediction])

        spans_sorted = sorted(list(span2prediction.keys()), key = lambda x:span2prediction[x][0][1])

        blocked = set()
        for start, end in spans_sorted:
            for x in range(start, end+1):
                if x in blocked:
                    del span2prediction[(start, end)]
                    break

            if (start, end) in span2prediction:
                blocked.update(range(start, end+1))

        spans = []
        for start, end in span2prediction:
            probas = np.array([x[1] for x in span2prediction[(start, end)]])
            ent = span2prediction[(start, end)][probas.argmax()][0]
            if ent is not None:
                spans.append([ent, start, end, probas.argmax()])

        return spans

    def predict_sentence(self, sentence, batch_size = 4, iterations = 1):#, gold_spans = None):
        sentence = copy.deepcopy(sentence)

        spans = []
        for it in range(iterations):
            pred_spans = self._predict_sentence(sentence, batch_size = batch_size)#, gold_spans = gold_spans)
            
            if len(pred_spans) == 0:
                break
            
            pred_spans.sort(key = lambda x:x[-1], reverse = True)

            if it + 1 < iterations:
                x = (it+1) * (len(spans) + len(pred_spans)) // iterations - len(spans)
                pred_spans = pred_spans[:max(x, 1)]
            
            spans.extend([x[:-1] for x in pred_spans]) 
            for entity, start, end, _ in pred_spans:
                assert not "[PAD]" in sentence[start:end+1]
                assert not any([x.startswith(self.ent_prefix) for x in sentence[start:end+1]])
                sentence = sentence[:start] + [self.ent_prefix + entity] + ["[PAD]"] * (end-start) + sentence[end+1:]
        

        return spans

    def predict_aida(self, in_file, out_file, batch_size = 4, iterations = 1):
        data = self.read_aida_file(in_file)#, ignore_gold = False)

        predictions = {}
        for idx, (sentence, _) in tqdm(list(data.items()), desc = "Prediction"):
            predictions[idx] = self.predict_sentence(sentence, batch_size = batch_size, iterations = iterations)#, gold_spans = None)

        norm_predictions = []
        offset = 0
        for key in sorted(list(predictions.keys())):
            predictions[key].sort(key = lambda x:x[1])
            for pred, start, end in predictions[key]:
                surface = " ".join(data[key][0][start:end+1])
                norm_predictions.append(f"{start+offset}\t{end+offset}\t{pred}\t{surface}\n")
            offset += len(data[key][0])            

        
        with open(out_file, "w") as whandle:
            whandle.write("".join(norm_predictions))


    def extract_spans(self, in_file, out_file):
        """
        Extract gold spans from AIDA file into a more handy format.
        
        in_file : File in standard AIDA format
        out_file : TSV with format start -- end -- entity (wiki) id -- surface form

        (Note that start and end positions are calculated w.r.t. to the entire file.)
        """
        counter = 0
        flag = False
        spans = []
        with open(in_file) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("DOCSTART") or line.startswith("DOCEND") or line == "*NL*" or len(line) == 0:
                    continue
                elif line.startswith("MMSTART"):
                    assert not flag
                    entity = line.strip().split()[-1]

                    entity = normalize_entity(self.ebert_emb, entity, self.ent_prefix)

                    spans.append([entity, [], counter, counter-1])
                    flag = True
                elif line.startswith("MMEND"):
                    flag = False
                elif flag:
                    spans[-1][-1] += 1
                    spans[-1][1].append(line)
                    counter += 1
                else:
                    counter += 1

        spans.sort(key = lambda x:(x[-2], x[-1]))

        with open(out_file, "w") as whandle:
            for entity, surface, start, end in spans:
                surface = " ".join(surface)
                whandle.write(f"{start}\t{end}\t{entity}\t{surface}\n")


    def train(self, train_file, dev_file, model_dir, test_file,
            wikidata_entity_types_path, use_type_emb, use_pos_emb, type_emb_option, use_ebert_emb,
            batch_size = 128, eval_batch_size = 4, gradient_accumulation_steps = 16, verbose = True,
            epochs = 15, warmup_proportion = 0.1, lr = 5e-5, do_reinit_lm = False, beta2 = 0.999,
            null_penalty=1.0):

        self.use_ebert_emb = use_ebert_emb
        self.type_emb_option = type_emb_option
        self.use_pos_emb = use_pos_emb
        self.model_dir = model_dir
        self.dev_file = dev_file
        self.test_file = test_file

        if use_type_emb and type_emb_option == 'mix':
            self.gate_hidden = nn.Linear(self.null_vector.shape[0],
                                         self.null_vector.shape[0])
            self.gate_hidden.to(device=self.device)


        # Create temp gold files for predicting later
        self.dev_gold_file = os.path.join(model_dir,
                             os.path.basename(self.dev_file) + ".gold.txt")
        self.test_gold_file = os.path.join(model_dir,
                              os.path.basename(self.test_file) + ".gold.txt")
        self.extract_spans(dev_file, self.dev_gold_file)
        self.extract_spans(test_file, self.test_gold_file)


        self.ent2idx = {None: 0}

        if do_reinit_lm:
            for module in self.model.cls.predictions.transform.modules():
                self.model._init_weights(module)

        if use_type_emb and not self.use_ebert_emb:
            self.wiki_title2qid, self.qid2label, self.qid2parent_qids = load_wikidata(wikidata_entity_types_path)

        train_data = self.read_aida_file(train_file, ignore_gold = False)
        dev_data = self.read_aida_file(dev_file, ignore_gold = False)
        test_data = self.read_aida_file(test_file, ignore_gold = True)

        #  self.printing_samples(train_data, "train.samples")
        #  self.printing_samples(dev_data, "dev.samples")
        #  test_data = self.read_aida_file(test_file, ignore_gold = False)
        #  self.printing_samples(test_data, "test.samples")

        train_samples = self.data2samples(train_data)

        # Count ratio of null label
        null_count = 0
        for sample in train_samples:
            correct_idx = sample.correct_idx
            if correct_idx == 0:
                null_count += 1
        null_ratio = null_count / len(train_samples)
        print(f"null_count:{null_count}/{len(train_samples)}, ratio:{null_ratio:.2f}")

        dev_samples = self.data2samples(dev_data)
        test_samples = self.data2samples(test_data)

        self.parameters = [self.null_vector]
        optimizer_grouped_parameters = [{'params': [self.null_vector], 'weight_decay': 0.01}, {'params': [], 'weight_decay': 0.0}]

        if self.do_use_priors:
            self.parameters.append(self.null_bias)
            optimizer_grouped_parameters[1]["params"].append(self.null_bias)

        for n, p in self.model.named_parameters():
            i = 1 if any([nd in n for nd in self.NO_DECAY]) else 0
            optimizer_grouped_parameters[i]['params'].append(p)
            self.parameters.append(p)


        num_train_steps_per_epoch = len(train_samples) // batch_size + int(len(train_samples) % batch_size != 0)
        num_train_steps = epochs * num_train_steps_per_epoch

        if beta2:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr = lr, betas=(0.9, beta2))
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr = lr)
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=warmup_proportion*num_train_steps, t_total=num_train_steps)

        valid_best_micro_f1 = -1
        best_epoch = 0
        # Record test scores when model performs best on valid set.
        test_micro_score = None
        test_macro_score = None
        #  self.save(model_dir, epoch = 0)

        for epoch in trange(epochs, desc = "Epoch", disable = not verbose):
            self.rnd.shuffle(train_samples)

            train_loss = self.train_loop(train_samples,
                    batch_size = batch_size // gradient_accumulation_steps,
                    gradient_accumulation_steps = gradient_accumulation_steps,
                    null_penalty=null_penalty)

            print(f"Finishied training epoch {epoch+1}, loss per sample: {train_loss}")
            micro_res, macro_res = self.get_predict_score(self.dev_file, self.dev_gold_file)
            print(f"valid micro p r f1: {micro_res} in epoch {epoch+1}")
            print(f"valid macro p r f1: {macro_res} in epoch {epoch+1}")
            if micro_res[2] > valid_best_micro_f1:
              valid_best_micro_f1 = micro_res[2]
              best_epoch = epoch
              print(f"new best score on valid in epoch {epoch+1} is achieved")
              test_micro_score, test_macro_score = self.get_predict_score(self.test_file, self.test_gold_file)

        print(f"In epoch {best_epoch+1}, the model performs best on valid set and the performance on test is as follows.")
        print(f"Test micro p r f1: {test_micro_score,}")
        print(f"Test macro p r f1: {test_macro_score}")


    def get_predict_score(self, file_path, file_gold_path):
        from score_aida import load_file
        from score_aida import score_micro_f1, score_macro_f1

        out_file_test = f"{os.path.basename(file_path)}.pred.txt"
        test_pred_file = os.path.join(self.model_dir, out_file_test)
        self.predict_aida(in_file = file_path, out_file = test_pred_file,
                batch_size = 4, iterations = 1)
        true = load_file(file_gold_path)
        pred = load_file(test_pred_file)
        micro_p, micro_r, micro_f1 = score_micro_f1(true, pred)
        macro_p, macro_r, macro_f1 = score_macro_f1(true, pred)
        return [micro_p, micro_r, micro_f1], [macro_p, macro_r, macro_f1]


    def get_ent_surface_emb(self, wiki_title):
      surface_emb = None
      wiki_title = wiki_title.replace('_', ' ')
      wiki_title = wiki_title.replace('(', '')
      wiki_title = wiki_title.replace(')', '')
      if wiki_title == '':
          return None
      wiki_title_tokenized = self.tokenizer.tokenize(wiki_title)
      wiki_title_ids = self.tokenizer.convert_tokens_to_ids(wiki_title_tokenized)
      wiki_title_ids = torch.tensor(wiki_title_ids)
      wiki_title_ids = wiki_title_ids.to(device=self.device)
      wiki_title_emb = self.bert_emb(wiki_title_ids) # shape: (n, 768)
      surface_emb = torch.mean(wiki_title_emb, dim=0) # shape: (768,)
      return surface_emb


    def get_ent_emb_from_type_knowledge(self, wiki_title):
      """"
      wiki_title: str, e.g., Jimmy_Wales
      return: tensor of shape (768,) if found else None
      """
      assert self.type_emb_option in ['type', 'surface', 'mix']
      type_emb = None
      if True:
        if (wiki_title in self.wiki_title2qid and
            self.wiki_title2qid[wiki_title] in self.qid2parent_qids):
            qid = self.wiki_title2qid[wiki_title]
            parent_qids = self.qid2parent_qids[qid]
            valid_parent_count = 0
            sum_label_emb = torch.zeros(size=(self.bert_emb.weight.shape[1],),
                                        dtype=self.bert_emb.weight.dtype)
            sum_label_emb = sum_label_emb.to(device=self.device)
            for qid in parent_qids:
                if qid in self.qid2label:
                    label = self.qid2label[qid]
                    label_tokenized = self.tokenizer.tokenize(label)
                    label_ids = self.tokenizer.convert_tokens_to_ids(label_tokenized)
                    label_ids = torch.tensor(label_ids).to(device=self.device)
                    label_emb = self.bert_emb(label_ids) # shape: (n, 768)
                    label_emb = torch.mean(label_emb, dim=0) # shape: (768,)
                    sum_label_emb = sum_label_emb + label_emb
                    valid_parent_count += 1
            if valid_parent_count > 0:
                mean_label_emb = sum_label_emb / valid_parent_count
                type_emb = mean_label_emb
      if type_emb is None:
          return self.get_ent_surface_emb(wiki_title)
      else:
          return type_emb


    def save(self, model_dir, epoch=None):
        f_model = os.path.join(model_dir, "model.pth") if epoch is None else os.path.join(model_dir, f"model_{epoch}.pth")
        f_null_vector = os.path.join(model_dir, "null_vector.pth") if epoch is None else os.path.join(model_dir, f"null_vector_{epoch}.pth")
        torch.save(self.model.state_dict(), f_model)
        torch.save(self.null_vector, f_null_vector)
        f_ent_emb = os.path.join(model_dir, "ent_emb.pth")
        f_ent2idx = os.path.join(model_dir, "ent2idx.pth")
        with open(f_ent2idx, 'wb') as fh:
            pickle.dump(self.ent2idx, fh)
        
        if self.do_use_priors:
            f_null_bias = os.path.join(model_dir, "null_bias.pth") if epoch is None else os.path.join(model_dir, f"null_bias_{epoch}.pth")
            torch.save(self.null_bias, f_null_bias)

    def load(self, model_dir, epoch = None):
        f_model = os.path.join(model_dir, "model.pth") if epoch is None else os.path.join(model_dir, f"model_{epoch}.pth")
        f_null_vector = os.path.join(model_dir, "null_vector.pth") if epoch is None else os.path.join(model_dir, f"null_vector_{epoch}.pth")
        f_ent_emb = os.path.join(model_dir, "ent_emb.pth")
        f_ent2idx = os.path.join(model_dir, "ent2idx.pth")
        self.model.load_state_dict(torch.load(f_model))
        self.null_vector.data = torch.load(f_null_vector).data
        with open(f_ent2idx, 'rb') as fh:
            self.ent2idx = pickle.load(fh)
        
        if self.do_use_priors:
            f_null_bias = os.path.join(model_dir, "null_bias.pth") if epoch is None else os.path.join(model_dir, f"null_bias_{epoch}.pth")
            self.null_bias.data = torch.load(f_null_bias).data

    def make_input_dict(self, samples):
        if self.dynamic_max_len:
            max_len = max([len(sample.tokenized) for sample in samples])
        else:
            max_len = self.max_len
        
        input_ids = torch.zeros((len(samples), max_len)).to(dtype = torch.long)
        input_ids = input_ids.to(device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        attention_mask = attention_mask.to(device=self.device)

        for i, sample in enumerate(samples):
            assert len(sample.tokenized) <= max_len
            assert len(sample.tokenized) == len(sample.input_ids)
            input_ids[i,:len(sample.input_ids)] = torch.tensor(sample.input_ids).to(dtype = input_ids.dtype)
            attention_mask[i,:len(sample.input_ids)] = 1

        if self.use_pos_emb:
            input_embeddings = self.bert_pos_emb(input_ids)
        else:
            input_embeddings = self.bert_emb(input_ids)
        #unk_id = self.tokenizer.vocab["[UNK]"]

        idx2ent = {self.ent2idx[ent]: ent for ent in self.ent2idx}
        for i, sample in enumerate(samples):
            for j, token in enumerate(sample.tokenized):
                if j == sample.mask_pos:
                    assert token == "[MASK]"
                    if self.do_prime_mask:
                        assert sample.candidates[0] is None
                        biases = [np.exp(x) for x in sample.biases[1:]]
                        weighted_ent = torch.zeros(size=(self.null_vector.shape[0],))
                        weighted_ent = weighted_ent.to(device=self.null_vector.device)
                        wiki_titles = [idx2ent[idx] for idx in sample.candidate_ids[1:]]
                        candidate_embeddings = [self.get_ent_emb_from_type_knowledge(wiki_title) for wiki_title in wiki_titles]
                        for biase, cand_emb in zip(biases, candidate_embeddings):
                            weighted_ent = weighted_ent + biase * cand_emb
                        input_embeddings[i,j,:] = weighted_ent.to(device=input_embeddings.device)
                        #  input_embeddings[i,j,:] = torch.tensor(np.mean(candidate_embeddings, 0)).to(dtype = input_embeddings.dtype)

                elif token.startswith(self.ent_prefix):
                    input_embeddings[i,j,:] = torch.tensor(self.ebert_emb[token]).to(dtype = input_embeddings.dtype)

        input_embeddings = input_embeddings.to(device = self.device)
        attention_mask = attention_mask.to(device = self.device)
        label_ids = torch.tensor([sample.correct_idx for sample in samples]).to(dtype = torch.long, device = self.device)

        return {"input_ids": input_embeddings, "attention_mask": attention_mask, "label_ids": label_ids}

    def train_loop(self, samples, batch_size, gradient_accumulation_steps,
                   verbose = True, null_penalty=1.0):
        return self.loop(samples = samples, batch_size = batch_size, 
                gradient_accumulation_steps = gradient_accumulation_steps, 
                mode = "train", verbose = verbose, null_penalty=null_penalty)

    def eval_loop(self, samples, batch_size, verbose = True):
        return self.loop(samples = samples, batch_size = batch_size, 
                gradient_accumulation_steps = 1, mode = "eval", verbose = verbose)
    
    def pred_loop(self, samples, batch_size, verbose = True):
        return self.loop(samples = samples, batch_size = batch_size, 
                gradient_accumulation_steps = 1, mode = "pred", verbose = verbose)

    def loop(self, samples, batch_size, mode, gradient_accumulation_steps,
             verbose = 1, null_penalty=1.0):
        assert mode in ("train", "eval", "pred")

        all_true, all_pred, all_losses, all_pred_spans = [], [], [], []

        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        idx2ent = {self.ent2idx[ent]: ent for ent in self.ent2idx}
        assert len(idx2ent) == len(self.ent2idx)
        assert sorted(list(idx2ent.keys())) == list(range(len(idx2ent)))
    
        for step, i in enumerate(trange(0, len(samples), batch_size, desc = f"Iterations ({mode})", disable = not verbose)):
            batch = samples[i:i+batch_size]

            all_true.extend([sample.correct_idx for sample in batch])
            mask_positions = [sample.mask_pos for sample in batch]

            input_dict = self.make_input_dict(batch)
            label_ids = input_dict.pop("label_ids")
        
            all_entities, ranges = [], []
            for j, sample in enumerate(batch):
                all_entities.extend(sample.candidate_ids[1:])
                ranges.append([len(all_entities) - len(sample.candidate_ids[1:]), len(all_entities)])
            
            all_outputs = self.model(**input_dict)[0]
            outputs = torch.stack([all_outputs[j, position] for j, position in enumerate(mask_positions)])


            wiki_titles = [idx2ent[idx] for idx in all_entities]
            wiki_title_type_embs = [self.get_ent_emb_from_type_knowledge(wiki_title) for wiki_title in wiki_titles]
            batch_entity_embedding = torch.stack(wiki_title_type_embs).to(device=self.device)

            wiki_title_surface_embs = [self.get_ent_surface_emb(wiki_title) for wiki_title in wiki_titles]
            batch_surface_embedding = torch.stack(wiki_title_surface_embs).to(device=self.device)
            gate_out = torch.sigmoid(self.gate_hidden(batch_entity_embedding))
            batch_entity_embedding = (1 - gate_out) * batch_entity_embedding  + gate_out * batch_surface_embedding

            batch_loss = 0
            for j, sample in enumerate(batch):
                assert len(sample.candidate_ids) == ranges[j][1] - ranges[j][0] + 1
                candidates_with_zero = torch.cat([self.null_vector.unsqueeze(0), batch_entity_embedding[ranges[j][0]:ranges[j][1]]])
                logits = candidates_with_zero.matmul(outputs[j])
                
                if self.do_use_priors:
                    assert sample.biases[0] is None
                    biases = torch.tensor(sample.biases[1:]).to(device = self.null_bias.device, dtype = self.null_bias.dtype)
                    # self.null_bias is for NONE entity category
                    # biases's shape: (len(entities), ) where entities are
                    # number of candidates for current span.
                    logits += torch.cat([self.null_bias, biases])
                
                probas = torch.softmax(logits, -1)

                if mode == "eval":
                    probas_numpy = probas.detach().cpu().numpy()
                    all_pred.append(probas_numpy.argmax())
            
                if mode == "pred":
                    probas_numpy = probas.detach().cpu().numpy()
                    entities = [None] + [idx2ent[i] for i in sample.candidate_ids[1:]]
                    all_pred_spans.append((entities[probas_numpy.argmax()], sample.data_idx, sample.start, sample.end, 
                        [(ent, float(p)) for ent, p in zip(entities, probas_numpy)]))

                elif mode == "train":
                    sample_loss = -torch.log(probas[label_ids[j]])
                    if label_ids[j] == 0:
                      sample_loss = sample_loss * null_penalty
                    batch_loss += sample_loss / len(batch)
                    all_losses.append(float(sample_loss.item()))

            if mode == "train":
                if (step+1) % 1000 == 0:
                    print(f"step {step+1}: mean loss over batch: {batch_loss.item()}")
                batch_loss.backward()
                
                if (step+1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
    
        if mode == "pred":
            return all_pred_spans
        if mode == "eval":
            return self.score_f1(all_true, all_pred)
        elif mode == "train":
            return np.mean(all_losses)


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


def train(args):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model_args = {"bert_name": args.bert_name, "ebert_name": args.ebert_name,
            "mapper_name": args.mapper_name,
            "granularity": args.granularity, "max_len": args.max_len,
            "max_candidates": args.max_candidates, "do_prime_mask": args.do_prime_mask,
            "do_use_priors": args.do_use_priors}

    train_args = {"train_file": args.train_file, "dev_file": args.dev_file,
            "test_file": args.test_file,
            "model_dir": args.model_dir, "lr": args.lr, "epochs": args.epochs,
            "do_reinit_lm": args.do_reinit_lm,
            "null_penalty": args.null_penalty,
            "type_emb_option": args.type_emb_option,
            "wikidata_entity_types_path": args.wikidata_entity_types_path,
            "use_pos_emb": args.use_pos_emb,
            "use_type_emb": args.use_type_emb,
            "use_ebert_emb": args.use_ebert_emb,
            "batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_proportion": args.warmup_proportion, "eval_batch_size": args.eval_batch_size}

    with open(os.path.join(args.model_dir, "model_args.json"), "w") as handle:
            json.dump(model_args, handle)
    with open(os.path.join(args.model_dir, "train_args.json"), "w") as handle:
            json.dump(train_args, handle)

    print(model_args, flush = True)
    print(train_args, flush = True)

    model = EntityLinkingAsLM(**model_args, device = args.device)
    model.train(**train_args)


if __name__ == "__main__":
    args = parse_args()
    print(os.uname(), flush = True)
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", None), flush = True)
    print(args, flush = True)

    train(args)
