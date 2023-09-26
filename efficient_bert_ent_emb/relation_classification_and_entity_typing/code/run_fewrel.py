# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import simplejson as json

#  from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as f1_score

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from knowledge_bert.tokenization import BertTokenizer
from knowledge_bert.modeling import BertForSequenceClassificationSplitDescrip
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from descrip_emb_util import load_descrip
from descrip_emb_util import split_ents
from descrip_emb_util import ResultRecorder

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_ent, ent_mask,
        label_id, target_ent, split_target_pos, target_ent_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_ent = input_ent
        self.ent_mask = ent_mask
        self.target_ent = target_ent
        self.split_target_pos = split_target_pos
        self.target_ent_mask = target_ent_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.loads(f.read())

class FewrelProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "dev")


    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
            text_a = (line['text'], line['ents'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
    tokenizer, thresholds, entity_id2parents, entity_id2label, max_parent,
    qid2idx, verbose=2):
    """Loads a data file into a list of `InputBatch`s."""
    
    label_list = sorted(label_list)
    label_map = {label : i for i, label in enumerate(label_list)}
    threshold, target_threshold = thresholds

    #  entity2id = {}
    #  with open("kg_embed/entity2id.txt") as fin:
        #  fin.readline()
        #  for line in fin:
            #  qid, eid = line.strip().split('\t')
            #  entity2id[qid] = int(eid)

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h, t = example.text_a[1]
        h_name = ex_text_a[h[1]:h[2]]
        t_name = ex_text_a[t[1]:t[2]]

        targets = [h, t]
        target_num = len(targets)
        #  ent_pos = [x for x in example.text_b if x[-1]>threshold]
        ent_pos = targets
        target_qids, non_target_qids = split_ents(ent_pos, targets, threshold, target_threshold)

        # Add [HD] and [TL], which are "#" and "$" respectively.
        if h[1] < t[1]:
            ex_text_a = ex_text_a[:h[1]] + "# "+h_name+" #" + ex_text_a[h[2]:t[1]] + "$ "+t_name+" $" + ex_text_a[t[2]:]
        else:
            ex_text_a = ex_text_a[:t[1]] + "$ "+t_name+" $" + ex_text_a[t[2]:h[1]] + "# "+h_name+" #" + ex_text_a[h[2]:]

        if h[1] < t[1]:
            h[1] += 2
            h[2] += 2
            t[1] += 6
            t[2] += 6
        else:
            h[1] += 6
            h[2] += 6
            t[1] += 2
            t[2] += 2
        #  tokens_a, entities_a = tokenizer.tokenize_with_descrip(ex_text_a, [h, t], entity_id2parents, entity_id2label, max_parent)
        tokens_a, split_target_ents, split_target_pos, entities_a = tokenizer.tokenize_with_split_descrip(
          ex_text_a, ent_pos, entity_id2parents, entity_id2label, target_qids,
          non_target_qids, max_parent)
        #  if len([x for x in entities_a if x!=["UNK"]*max_parent]) != 2:
            #  print(f"QID do not have two for fewrel")
            #  exit(1)

        tokens_b = None
        if example.text_b:
            tokens_b, entities_b = tokenizer.tokenize(example.text_b[0], [x for x in example.text_b[1] if x[-1]>threshold])
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, entities_a, entities_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2" and target_num
            if len(tokens_a) > max_seq_length - 2 - target_num:
                tokens_a = tokens_a[:(max_seq_length - 2 - target_num)]
                entities_a = entities_a[:(max_seq_length - 2 - target_num)]
            if len(split_target_ents) > target_num:
              split_target_ents = split_target_ents[:target_num]
              split_target_pos = split_target_pos[:target_num]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] * (target_num + 1)
        ents = [["UNK"]*max_parent] + entities_a + [["UNK"]*max_parent] * (target_num + 1)
        segment_ids = [0] * len(tokens)
        # Update split_target_pos
        for i in range(len(split_target_pos)):
          split_target_pos[i] += 1

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            ents += entities_b + ["UNK"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ent = []
        ent_mask = []
        for ent in ents:
            input_ent_ = []
            ent_mask_ = []
            for qid in ent:
              if qid != "UNK" and qid in qid2idx:
                  input_ent_.append(qid2idx[qid])
                  ent_mask_.append(1)
              else:
                  input_ent_.append(0)
                  ent_mask_.append(0)
            input_ent.append(input_ent_)
            ent_mask.append(ent_mask_)
        #  ent_mask[0] = 1

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Change extra target_num to 0
        for i in range(target_num):
          input_mask[-(i+1)] = 0

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # Change left target_num mask
        if len(split_target_pos) == 1:
          input_mask[-2] = 1
        if len(split_target_pos) == 2:
          input_mask[-2] = 1
          input_mask[-1] = 1

        # Padding split_target_pos
        padding = [0] * (target_num - len(split_target_pos))
        split_target_pos += padding
        target_ent = []
        target_ent_mask = []
        for ent in split_target_ents:
            target_ent_ = []
            target_ent_mask_ = []
            for qid in ent:
              if qid != "UNK" and qid in qid2idx:
                  target_ent_.append(qid2idx[qid])
                  target_ent_mask_.append(1)
              else:
                  target_ent_.append(0)
                  target_ent_mask_.append(0)
            target_ent.append(target_ent_)
            target_ent_mask.append(target_ent_mask_)

        padding = [[0]*max_parent] * (target_num - len(target_ent))
        target_ent += padding
        target_ent_mask += padding

        padding = [[0]*max_parent] * (max_seq_length - len(input_ent))
        ent_mask += padding
        input_ent += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_ent) == max_seq_length
        assert len(ent_mask) == max_seq_length
        assert len(split_target_pos) == target_num
        assert len(target_ent) == target_num
        assert len(target_ent_mask) == target_num

        label_id = label_map[example.label]
        if ex_index <  verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("ents: %s" % " ".join(
                    [str(x) for x in ents]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info(f"target_ent: {target_ent}")
            logger.info(f"target_ent_mask: {target_ent_mask}")
            logger.info(f"split_target_pos: {split_target_pos}")

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              input_ent=input_ent,
                              ent_mask=ent_mask,
                              label_id=label_id,
                              target_ent=target_ent,
                              split_target_pos=split_target_pos,
                              target_ent_mask=target_ent_mask))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, ents_a, ents_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ents_a.pop()
        else:
            tokens_b.pop()
            ents_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--no_descrip",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--emb_base",
                        default="descrip",
                        type=str)
    parser.add_argument("--entities_tsv", default=None, type=str, required=True,
                        help="entties files where descriptions are stored.")
    parser.add_argument("--sort",
                        default="long",
                        type=str, help="short, long and random to sort descrip.")
    parser.add_argument("--max_parent",
                        default=5,
                        type=int)
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ernie_model", default=None, type=str, required=True,
                        help="Ernie pre-trained model")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=234,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--threshold', type=float, default=.0)
    parser.add_argument('--target_threshold', type=float, default=.0)
    parser.add_argument('--note', type=str, default="")

    args = parser.parse_args()
    rr = ResultRecorder(note=args.note)

    entity_id2label, entity_id2parents, qid2idx, descrip_embs = load_descrip(args.emb_base, args.entities_tsv)
    processors = FewrelProcessor

    num_labels_task = 80

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    #  if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        #  raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    #  os.makedirs(args.output_dir, exist_ok=True)


    processor = processors()
    num_labels = num_labels_task
    label_list = None

    tokenizer = BertTokenizer.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case, sort=args.sort)

    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model, _ = BertForSequenceClassificationSplitDescrip.from_pretrained(args.ernie_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
              num_labels = num_labels, descrip_embs=descrip_embs.cuda())
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    global_step = 0
    

    def do_eval(mode="test", current_step=None):
      dev_examples = processor.get_dev_examples(args.data_dir)
      dev = convert_examples_to_features(
          dev_examples, label_list, args.max_seq_length, tokenizer,
          [args.threshold, args.target_threshold], entity_id2parents, entity_id2label,
          args.max_parent, qid2idx, 1)

      test_examples = processor.get_test_examples(args.data_dir)
      test = convert_examples_to_features(
          test_examples, label_list, args.max_seq_length, tokenizer,
          [args.threshold, args.target_threshold], entity_id2parents, entity_id2label,
          args.max_parent, qid2idx, 1)

      if mode == "dev":
          eval_features = dev
          eval_examples = dev_examples
      else:
          eval_features = test
          eval_examples = test_examples

      all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
      all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
      all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
      all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
      all_ent = torch.tensor([f.input_ent for f in eval_features], dtype=torch.long)
      all_ent_masks = torch.tensor([f.ent_mask for f in eval_features], dtype=torch.long)
      all_target_ent = torch.tensor([f.target_ent for f in eval_features], dtype=torch.long)
      all_target_pos = torch.tensor([f.split_target_pos for f in eval_features], dtype=torch.long)
      all_target_ent_mask = torch.tensor([f.target_ent_mask for f in
        eval_features], dtype=torch.long)
      eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
          all_ent, all_ent_masks, all_label_ids, all_target_ent,
          all_target_pos, all_target_ent_mask)
      # Run prediction for full data
      eval_sampler = SequentialSampler(eval_data)
      eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

      model.eval()
      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0
      eval_label_ids = []
      eval_preds = []
      for input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids, target_ent, split_target_pos, target_ent_mask in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        input_ent = input_ent.to(device)
        ent_mask = ent_mask.to(device)
        label_ids = label_ids.to(device)
        target_ent = target_ent.to(device)
        split_target_pos = split_target_pos.to(device)
        target_ent_mask = target_ent_mask.to(device)

        with torch.no_grad():
          tmp_eval_loss = model(input_ids, segment_ids, input_mask, input_ent,
              ent_mask, label_ids, tokenizer=tokenizer, qid2idx=qid2idx,
              entity_id2label=entity_id2label, use_ent_emb=(not
                args.no_descrip), target_ent=target_ent,
              split_target_pos=split_target_pos,
              target_ent_mask=target_ent_mask)
          logits = model(input_ids, segment_ids, input_mask, input_ent,
              ent_mask, tokenizer=tokenizer, qid2idx=qid2idx,
              entity_id2label=entity_id2label, use_ent_emb=(not
                args.no_descrip), target_ent=target_ent,
              split_target_pos=split_target_pos,
              target_ent_mask=target_ent_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy, pred = accuracy(logits, label_ids)
        eval_preds.extend(pred.tolist())
        eval_label_ids.extend(label_ids.tolist())

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

      eval_loss = eval_loss / nb_eval_steps
      eval_accuracy = eval_accuracy / nb_eval_examples

      p, r, f, _ = f1_score(y_true=eval_label_ids, y_pred=eval_preds, average='micro')
      result = {'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy ,
                'p': p, 'r':r, 'f': f
                }
      rr.record(current_step, [p, r, f], mode)
      return result


    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer,
            [args.threshold, args.target_threshold], entity_id2parents, entity_id2label,
            args.max_parent, qid2idx)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_ent = torch.tensor([f.input_ent for f in train_features], dtype=torch.long)
        all_ent_masks = torch.tensor([f.ent_mask for f in train_features], dtype=torch.long)
        all_target_ent = torch.tensor([f.target_ent for f in train_features], dtype=torch.long)
        all_target_pos = torch.tensor([f.split_target_pos for f in train_features], dtype=torch.long)
        all_target_ent_mask = torch.tensor([f.target_ent_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask,
            all_segment_ids, all_ent, all_ent_masks, all_label_ids,
            all_target_ent, all_target_pos, all_target_ent_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        output_loss_file = os.path.join(args.output_dir, "loss")
        #  loss_fout = open(output_loss_file, 'w')
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for i, t in enumerate(batch))
                #  input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids = batch
                input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids, target_ent, split_target_pos, target_ent_mask = batch
                #  input_ent = embed(input_ent+1).to(device) # -1 -> 0
                #  loss = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, label_ids)
                loss = model(input_ids, segment_ids, input_mask, input_ent,
                    ent_mask, label_ids, tokenizer=tokenizer, qid2idx=qid2idx,
                    entity_id2label=entity_id2label, use_ent_emb=(not
                      args.no_descrip), target_ent=target_ent,
                    split_target_pos=split_target_pos,
                    target_ent_mask=target_ent_mask)
                #  loss = model(input_ids, segment_ids, input_mask, input_ent.half(), ent_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                #  loss_fout.write("{}\n".format(loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            result = do_eval("dev", epoch+1)
            logger.info(f"*****Dev result@epoch {epoch+1}: {result} *****")
            # Record test scores in rr and the one with best dev score will be printed.
            do_eval("test", epoch+1)
            model.train()

        rr.print()
        # Save a trained model
        #  model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        #  output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        #  torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    main()
