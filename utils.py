# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import tarfile
import tempfile

import torch

from pytorch_pretrained_bert import cached_path
from collections import defaultdict

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

logger = logging.getLogger(__file__)

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
QN_WORDS = ['who', 'what', 'where', 'why', 'when', 'how', 'which', 'whom', 'whose', '?']
lm_special_tokens = ['_start_', '_delimiter_', '_classify_']

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL,cache_dir='./cache/')
    tempdir = tempfile.mkdtemp()

    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def get_dataset(tokenizer, dataset_path, dataset_cache=None):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path,cache_dir='./cache/')
        # personachat_file = cached_path(dataset_path, cache_dir='../../.pytorch_pretrained_bert')
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    # FIXME: only for testing, delete later:
    '''
    dataset['train'] = dataset['train'][:1]
    while len(dataset['train'][0]['utterances']) != 1:
        dataset['train'][0]['utterances'].pop()
    # dataset['valid'] = dataset['valid'][:1]
    dataset['valid'] = dataset['train']
    '''
    dataset['train'] = dataset['train'][:int(len(dataset['train'])*0.9)]
    # dataset['train'] = dataset['train'][:int(len(dataset['train']) * 0.1)]

    # train_len = int(len(dataset['train']) * 0.9)
    # dataset['train'] = dataset['train'][int(len(dataset['train']) * 0.9):]

    # dataset['train'] = dataset['train'][: 1]
    dataset['valid'] = dataset['valid'][: 1]   # 这里不要乱改啊！！！
    # dataset['train'] = dataset['train'][:int(len(dataset['train'])*0.9)]
    # dataset['dev'] = dataset['train'][int(len(dataset['train']) * 0.9):]

    personachat_file = cached_path(dataset_path,cache_dir='./cache/')
    # personachat_file = cached_path(dataset_path, cache_dir='../../.pytorch_pretrained_bert')
    with open(personachat_file, "r", encoding="utf-8") as f:
        org_dataset = json.loads(f.read())
    # org_dataset_tmp = org_dataset['train'][train_len:]
    # personas = defaultdict(list)
    for dataset_name in org_dataset:
        for i, dialogue in enumerate(org_dataset[dataset_name]):
            if i >= len(dataset[dataset_name]):
                break
            dataset[dataset_name][i]['persona_org'] = dialogue['personality'].copy()
            '''
            for _ in range(len(dialogue['utterances'])):
                personas[dataset_name].append(dialogue['personality'])
                '''
    return dataset

def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path,cache_dir='./cache/')
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())
            print(personachat)

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        personachat = tokenize(personachat)
        torch.save(personachat, dataset_cache)

    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])



    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    # YW: speaker1 is user
    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]   # YW: add speaker infomation

    instance["input_ids"] = list(chain(*sequence))   # YW: concat all the context
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]   # YW: TODO: persona is speaker1?
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])   # YW: all -1 (mask?) -> because it's not the right candidate(label)!
    if lm_labels:   # YW: if it's label
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence