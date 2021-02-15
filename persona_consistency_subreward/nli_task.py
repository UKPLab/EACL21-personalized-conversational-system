# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam   # warmup_linear
import jsonlines
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        opts:
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

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.label_id = label_id


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
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    def _read_jsonl(cls, input_file):
        lines = []
        lines.append(['id', 'sentence1', 'sentence2', 'label'])
        data = jsonlines.Reader(open(input_file, 'r', encoding='UTF-8')).read()
        for sample in data:
            lines.append([sample['id'], sample['sentence1'], sample['sentence2'], sample['label']])   # TODO: only 'e2swap_up'
        return lines



class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_dev_examples(self, pred, profiles):
        """See base class."""
        return self._create_examples(pred, profiles)

    def get_labels(self):
        """See base class."""
        return ["negative", "positive", "neutral"]

    def _create_examples(self, pred, profiles):
        """Creates examples for the training and dev sets."""   # TODO: now just for one prediction
        examples = []
        for (i, line) in enumerate(profiles):
            guid = "%s-pair" % (i)
            text_a = pred
            text_b = line
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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
        else:
            tokens_b.pop()


def score_softmax(out):
    """Reward score calculation with Softmax function."""
    outputs = F.softmax(out, dim=1)
    negative = outputs.detach().cpu().numpy()[:, 0]
    positive = outputs.detach().cpu().numpy()[:, 1]
    neutral = outputs.detach().cpu().numpy()[:, 2]
    res = 1 * positive + 0 * neutral - 2 * negative
    return res


def score_softmax_allres(out):
    """Reward score calculation for all previous responses (not applied)."""
    outputs = F.softmax(out, dim=1)
    negative = outputs.detach().cpu().numpy()[:, 0]
    positive = outputs.detach().cpu().numpy()[:, 1]
    neutral = outputs.detach().cpu().numpy()[:, 2]
    res = 0 * positive + 0 * neutral - 2 * negative
    return res


def score(out):
    out = out.cpu().detach().numpy()
    outputs = np.argmax(out, axis=1)
    res = []
    for e in outputs:
        if e == 0:
            res.append(-1)   # -1
        elif e == 1:
            res.append(1)   # 1
        elif e == 2:
            res.append(0)   # 0
    return np.array(res)


def nli_score(out):
    """Get the NLI score from logit."""
    out = out.cpu().detach().numpy()
    outputs = np.argmax(out, axis=1)
    res = np.array([0, 0, 0])
    for e in outputs:
        if e == 0:
            res[0] += 1   # -1 for negative (contradiction)
        elif e == 1:
            res[1] += 1   # 1 for positive (entailment)
        elif e == 2:
            res[2] += 1   # 0 for neutral
    c_score = (-1 * res[0] + 1 * res[1] + 0 * res[2]) / out.shape[0]
    con_en = np.array([res[0] > 0, res[1] > 0, res[2] > 0]) + 0   # if the persona contains at least one contradict/entail/neutral profile
    return np.array(res), c_score, con_en


def main(pred, profiles, tokenizer=None, model=None, eval=False, allres=False):
    """
    Use an NLI engine in reward function.
    :param pred: the predicted response.
    :param profiles: the input profiles.
    :param tokenizer: BERT tokenizer.
    :param model: BERT model.
    :param eval: eval mode for evaluation phase (convai_evaluation_edit_reward.py).
    :param allres: previous response mode (not applied now).
    :return: NLI score.
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
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
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    opts = parser.parse_args(['--task_name=MNLI',
                              '--do_lower_case',
                              '--data_dir=../data/dialogue_nli',   # "../data/dialogue_nli_test" has less data for code testing.
                              '--bert_model=bert-base-uncased',
                              '--max_seq_length=128',
                              '--output_dir=nli_output/'])

    if opts.server_ip and opts.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(opts.server_ip, opts.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "mnli": MnliProcessor,
    }

    num_labels_task = {
        "mnli": 3,
    }

    if opts.local_rank == -1 or opts.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(opts.local_rank)
        device = torch.device("cuda", opts.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            opts.gradient_accumulation_steps))

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

    task_name = opts.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    if (tokenizer is None) and (model is None):
        raise Exception('Tokenizer or model is not loaded.')

    if opts.local_rank == -1 or torch.distributed.get_rank() == 0:
        eval_examples = processor.get_dev_examples(pred, profiles)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, opts.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=opts.eval_batch_size)

        if not eval:

            reward_score = []

            for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):

                input_ids = input_ids.to(opts.device)
                input_mask = input_mask.to(opts.device)
                segment_ids = segment_ids.to(opts.device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)

                if allres:
                    reward_score.extend(score_softmax_allres(logits))
                else:
                    reward_score.extend(score_softmax(logits))
                # _, c_score = nli_score(logits)
                # reward_score.append(c_score)
            # print('INFO - nli - current reward:', sum(reward_score))
            return reward_score
        else:
            nli_scores_sum = [0, 0, 0]
            reward_score = []
            c_scores = 0

            for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):

                input_ids = input_ids.to(opts.device)
                input_mask = input_mask.to(opts.device)
                segment_ids = segment_ids.to(opts.device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)

                current_nli_score, current_c_score, con_en = nli_score(logits)
                nli_scores_sum += current_nli_score
                c_scores += current_c_score
                reward_score.extend(score_softmax(logits))
                print('\n\rINFO - nli - current reward:', sum(reward_score))
            return nli_scores_sum, sum(reward_score), c_scores, con_en
