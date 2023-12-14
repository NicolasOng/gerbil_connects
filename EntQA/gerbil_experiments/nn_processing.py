import torch
import torch.nn as nn
import numpy as np
from utils import Logger
from transformers import BertTokenizer
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../')))
from run_retriever import load_model as load_retriever
from run_reader import load_model as load_reader
from data_retriever import get_embeddings, \
    get_hard_negative
from reader import prune_predicts
from gerbil_experiments.data import get_retriever_loader, get_reader_loader, \
    load_entities, \
    get_reader_input, process_raw_data,  \
    get_doc_level_predicts, token_span_to_gerbil_span, \
    get_raw_results, process_raw_predicts


class Annotator(object):

    def __init__(self, args):
        self.args = args
        self.logger = self.set_logger()
        self.all_cands_embeds, self.entities = self.load_cands_part()
        #print(len(self.all_cands_embeds))
        #print(len(self.entities))
        #print(self.all_cands_embeds[0])
        #print(self.all_cands_embeds[-1])
        #print(self.all_cands_embeds.shape)
        self.my_device = torch.device('cuda' if torch.cuda.is_available()
                                      else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.dp = torch.cuda.device_count() > 1
        self.config = {
            "top_k": 100,
            "biencoder_model": self.args.blink_dir + "biencoder_wiki_large.bin",
            "biencoder_config": self.args.blink_dir + "biencoder_wiki_large.json"
        }
        self.model_retriever = load_retriever(False,
                                              self.config['biencoder_config'],
                                              self.args.retriever_path,
                                              self.my_device,
                                              self.args.type_retriever_loss,
                                              True)

        if self.dp:
            self.logger.log('Data parallel across {:d} GPUs {:s}'
                            ''.format(len(self.args.gpus.split(',')),
                                      self.args.gpus))
            self.model_retriever = nn.DataParallel(self.model_retriever)
        self.model_retriever.to(self.my_device)
        self.model_retriever.eval()
        self.model_reader = load_reader(False, self.args.reader_path,
                                        self.args.type_encoder,
                                        self.my_device,
                                        self.args.type_span_loss,
                                        self.args.do_rerank,
                                        self.args.type_rank_loss,
                                        self.args.max_answer_len,
                                        self.args.max_passage_len)
        self.model_reader.to(self.my_device)
        if self.dp:
            self.logger.log('Data parallel across %d GPUs: %s' %
                            (len(self.args.gpus.split(',')), self.args.gpus))
            self.model_reader = nn.DataParallel(self.model_reader)
        
        self.use_alt_candidate_set = False
        self.alt_candidate_set = None

    def set_empty1_setting(self):
        self.entities = []
        self.all_cands_embeds = []

    def set_empty2_setting(self):
        self.use_alt_candidate_set = True
        self.alt_candidate_set = []

    def set_full1_setting(self):
        # 1. load the full candidate set
        import pickle
        with open("candidate_list.pkl", 'rb') as file:
            full_candidate_set = pickle.load(file)
        # 2. replace the entities and embeddings list
        replacement_entities = []
        replacement_all_cands_embeds = []
        for title in full_candidate_set:
            for i, entity in enumerate(self.entities):
                if entity['title'] == title:
                    replacement_entities.append(self.entities[i])
                    replacement_all_cands_embeds.append(self.all_cands_embeds[i])
                    break
        self.entities = replacement_entities
        self.all_cands_embeds = np.array(replacement_all_cands_embeds)

    def set_full2_setting(self):
        self.use_alt_candidate_set = True
        # 1. load the full candidate set
        import pickle
        with open("candidate_list.pkl", 'rb') as file:
            full_candidate_set = pickle.load(file)
        # 2. search for the candidate in self.entities, and save its index.
        self.alt_candidate_set = []
        for title in full_candidate_set:
            for i, entity in enumerate(self.entities):
                if entity['title'] == title:
                    self.alt_candidate_set.append(i)
                    break

    def set_logger(self):
        logger = Logger(self.args.log_path + '.log', True)
        logger.log(str(self.args))
        return logger

    def load_cands_part(self):
        self.logger.log('load all candidates embeds')
        all_cands_embeds = np.load(self.args.cands_embeds_path)
        self.logger.log('load entities')
        entities = load_entities(self.args.ents_path)
        return all_cands_embeds, entities

    def get_predicts(self, document):
        samples_retriever, token2char_start, \
        token2char_end = process_raw_data(document,
                                          self.tokenizer,
                                          self.args.passage_len,
                                          self.args.stride)
        retriever_loader = get_retriever_loader(samples_retriever,
                                                self.tokenizer,
                                                self.args.bsz_retriever,
                                                self.args.max_len_retriever,
                                                self.args.add_topic,
                                                self.args.use_title)
        test_mention_embeds = get_embeddings(retriever_loader,
                                             self.model_retriever, True,
                                             self.my_device)
        top_k_test, scores_k_test = get_hard_negative(test_mention_embeds,
                                                      self.all_cands_embeds,
                                                      self.args.k, 0, False)
        #print(top_k_test[:2])
        #print(samples_retriever[:2])
        #print(self.entities[:2])
        self.logger.log('reader part')
        samples_reader = get_reader_input(samples_retriever, top_k_test,
                                          self.entities, alt_candidate_set=self.alt_candidate_set)
        #print(samples_reader[0])
        reader_loader = get_reader_loader(samples_reader, self.tokenizer,
                                          self.args.max_len_reader,
                                          self.args.max_num_candidates,
                                          self.args.bsz_reader,
                                          self.args.add_topic,
                                          self.args.use_title)
        raw_predicts = get_raw_results(self.model_reader, self.my_device,
                                       reader_loader, self.args.num_spans,
                                       samples_reader, self.args.do_rerank,
                                       True, self.args.no_multi_ents)
        #print(len(raw_predicts))
        #print(raw_predicts[0].shape)
        pruned_predicts = prune_predicts(raw_predicts, self.args.thresd)
        #print(pruned_predicts)
        transformed_predicts = process_raw_predicts(pruned_predicts,
                                                    samples_reader)
        #print(transformed_predicts)
        doc_predicts_span = get_doc_level_predicts(transformed_predicts,
                                                   self.args.stride)
        #print(doc_predicts_span)
        doc_predicts_gerbil = token_span_to_gerbil_span(doc_predicts_span,
                                                        token2char_start,
                                                        token2char_end)
        #print(doc_predicts_gerbil)
        return doc_predicts_gerbil
