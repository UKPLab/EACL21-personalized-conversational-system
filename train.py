# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict, Counter
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_pretrained_bert import OpenAIAdam, OpenAIGPTTokenizer, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME, BertModel, OpenAIGPTLMHeadModel
from modeling_openai import OpenAIGPTDoubleHeadsModel
from modeling_gpt2 import GPT2DoubleHeadsModel
from persona_consistency_subreward.nli_task import main as nli_engine
#from lib.bert_cls.nli_task import main as nli_engine
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer

from pytorch_pretrained_bert.modeling_openai import OpenAIGPTConfig

import time
import numpy as np
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_dataset, QN_WORDS
from gen_utils import sample_sequence
from rl_utils import create_critic, f1_rewarder, bleu_rewarder, plot_reward, LinearRegressionModel, process_document, \
    init_stop_words, read_model, tokens_to_vector, bert_vector
from rep_utils import get_ngrams, intrep_frac, flatten, extrep_frac
import traceback
import itertools
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids", "sample_index"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

# TODO: training shuffle, utils' training percentage - now off

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


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


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)   # YW: 'train': 17878; 'valid': 1000

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)

        personas = []   # YW
        personas_org = []
        turns = []
        labels = []
        sample_ind = 0

        for dialog in dataset:
            persona = dialog["personality"].copy()
            persona_org = dialog['persona_org'].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance, _ = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                    datasets[dataset_name]["sample_index"].append(sample_ind)
                    sample_ind += 1
                    personas.append(persona)
                    personas_org.append(persona_org)
                    # personas.append(persona)   # YW
                    # turns.append(input_array)
                    turns.append(history)
                    labels.append(utterance['candidates'][-1])
                persona = [persona[-1]] + persona[:-1]  # permuted personalities   # FIXME: why permuted personalities?

        datasets[dataset_name]['personas'] = personas   # YW
        datasets[dataset_name]['personas_org'] = personas_org
        datasets[dataset_name]['turns'] = turns
        datasets[dataset_name]['labels'] = labels

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:   # YW: make sure that the data are in MODEL_INPUTS
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels" and input_name != "sample_index":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    # train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler, datasets


def generate(persona, history, tokenizer, model, args):
    with torch.no_grad():
        out_ids = sample_sequence(persona, history, tokenizer, model, args)
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out_text, out_ids


def cos_sim(s1, s2, vocab_size):
    word_vector1 = np.zeros(vocab_size)
    word_vector2 = np.zeros(vocab_size)
    count1 = Counter(s1)
    count2 = Counter(s2)
    for item in count1.items():
        if item[0] <= vocab_size:
            word_vector1[item[0]] = item[1]
    for item in count2.items():
        if item[0] <= vocab_size:
            word_vector2[item[0]] = item[1]
    cos_sim = cosine_similarity(word_vector1.reshape(1, -1), word_vector2.reshape(1, -1))
    return cos_sim[0][0]


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    parser = ArgumentParser()
    # Transfertransfo
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="trained_agents/Nov09_21-11-05_krusty/", help="Path, url or short name of the model")   # openai-gpt   runs/Jun03_00-25-57_krusty/   runs/Aug05_20-02-37_krusty_2ep/
    parser.add_argument("--num_c"
                        "andidates", type=int, default=2, help="Number of candidates for training")   # 2
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")   # 2
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")   # 4
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")   # 4
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")   # 8
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")   # 6.25e-5
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")   # 3
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

    # Generation
    parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--no_sample", action='store_true')
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--min_length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)  # 0
    parser.add_argument("--temperature", type=int, default=0.7)
    parser.add_argument("--top_k", type=int, default=0)  # 20
    parser.add_argument("--top_p", type=float, default=0.9)  # del

    # RL - Critic
    parser.add_argument("-critic_pretrain_epochs", type=int, default=0, help="Number of epochs to pretrain critic (actor fixed).")
    parser.add_argument("-reinforce_lr", type=float, default=6.25e-5, help="""Learning rate for reinforcement training.""")
    parser.add_argument("-end_epoch", type=int, default=10, help="Epoch to stop training.")   # Default: 50
    parser.add_argument("-plot_interval", type=int, default=10, help="Bach interval to plot reward.")  # Default: 50
    parser.add_argument("-nli_reward", type=bool, default=True, help="Use persona NLI in reward function (model1).")   # persona NLI sub-reward
    parser.add_argument("-nli_weight", type=float, default=0.45, help="Weight of persona NLI.")
    parser.add_argument("-nli_allres_reward", type=bool, default=True, help="Use persona NLI in reward function (model1).")   # previous responses NLI sub-reward (not applied)
    parser.add_argument("-nli_allres_weight", type=float, default=1, help="Weight of response NLI.")
    parser.add_argument("-cos_sim_bert_reward", type=bool, default=True, help="Use cosine similarity (bert) in reward function.")   # cosine similarity sub-reward (with the last response from partner's side)
    parser.add_argument("-cos_sim_bert_weight", type=float, default=0.1, help="Weight of BERT based cosine similarity.")
    parser.add_argument("-intern_rep_reward", type=bool, default=True, help="Use internal repeatition in reward function.")   # internal repetition sub-reward
    parser.add_argument("-intern_rep_weight", type=float, default=0.2, help="Weight of internal repetition.")
    parser.add_argument("-extern_rep_reward", type=bool, default=False, help="Use external repeatition in reward function.")   # external repetition sub-reward (not applied)
    parser.add_argument("-lm_reward", type=bool, default=True, help="Use LM ppl in reward function.")   # fine-tuned GPT-based language model sub-reward
    parser.add_argument("-lm_weight", type=float, default=0.2, help="Weight of language model.")
    parser.add_argument("-qback_reward", type=bool, default=True, help="Use question back in reward function.")   # question-back sub-reward (not applied)
    parser.add_argument("-qback_weight", type=float, default=0.05, help="Weight of question back.")
    parser.add_argument("-f1_reward", type=bool, default=False, help="Use F1 score in reward function.")   # F1 score as reward function
    parser.add_argument("-bleu_reward", type=bool, default=False, help="Use BLEU in reward function.")   # BLEU score as reward function
    parser.add_argument("-critic_train", type=int, default=5, help="How many times the critic will be trained in each batch.")

    # GPU
    parser.add_argument("-log_interval", type=int, default=100, help="Print stats at this interval.")

    # NLI
    parser.add_argument("--do_lower_case", type=bool, default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--output_dir", default='./persona_consistency_subreward/nli_output/', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    reset_seed(args.seed)
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, cache_dir='./tmp/')
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint, cache_dir='./tmp/')
    tokenizer.set_special_tokens(SPECIAL_TOKENS)
    model.set_num_special_tokens(len(SPECIAL_TOKENS))
    model.to(args.device)
    optimizer = OpenAIAdam(model.parameters(), lr=args.lr)   # TODO: tbd


    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler, datasets = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def rl_update(engine, batch):
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        # RL training starts here
        model.eval()

        # Get all information in the current batch for reconstruction
        batch_dict = {}
        batch_dict['input_ids'] = batch[0].cpu().detach().numpy()
        batch_dict['mc_token_ids'] = batch[1].cpu().detach().numpy()
        batch_dict['lm_labels'] = batch[2].cpu().detach().numpy()
        batch_dict['token_type_ids'] = batch[4].cpu().detach().numpy()

        input_ids = batch[0].cpu().detach().numpy().tolist()
        mc_token_ids = batch[1].cpu().detach().numpy().tolist()
        lm_labels = batch[2].cpu().detach().numpy().tolist()
        token_type_ids = batch[4].cpu().detach().numpy().tolist()

        rewards = []
        persona_rewards = []
        response_rewards = []
        cos_sim_bert_rewards = []
        intern_rep_rewards = []
        extern_rep_rewards = []
        lm_rewards = []
        qback_rewards = []
        f1_rewards = []
        bleu_rewards = []

        batch_size = 0
        for i in range(len(batch[5])):
            sample_index = batch[5][i]
            personality = datasets['train']['personas'][sample_index]
            history = datasets['train']['turns'][sample_index]
            rl_train_personas_org = datasets['train']['personas_org'][sample_index]   # get the original persona text
            response, response_ids = generate(personality, history, tokenizer, model, args)   # get response text and tokenized response

            response_org = response
            response = response.replace(' \' ', '\'')   # TODO: tbd
            response_ids.append(tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[1]))   # add eos
            logger.info("rl_trainer - current response: %s", response)

            persona_reward = 0
            response_reward = 0
            cos_sim_bert_reward = 0
            intern_rep_reward = 0
            extern_rep_reward = 0
            lm_reward = 0
            qback_reward = 0
            f1_reward = 0
            bleu_reward = 0

            # persona NLI
            if args.nli_reward:
                scores = nli_engine(response, rl_train_personas_org, nli_tokenizer, nli_model)
                current_persona_reward = sum(scores) / len(rl_train_personas_org)
                logger.info('persona_reward = %f', current_persona_reward)
                persona_reward = current_persona_reward

            # previous responses NLI
            if args.nli_allres_reward:
                # history_chain = list(chain(*history))
                # history_text = tokenizer.decode(history_chain, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pre_responses = []
                for j in range(-len(history), 0):
                    if j % 2 == 0:
                        current_text = tokenizer.decode(history[j], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        pre_responses.append(current_text)
                response_scores = nli_engine(response, pre_responses, nli_tokenizer, nli_model, allres=True)
                if response_scores == []:
                    current_response_reward = 0
                else:
                    current_response_reward = sum(response_scores) / len(response_scores)
                logger.info('response_reward = %f', current_response_reward)
                response_reward = current_response_reward

            # cosine similarity NLI (with the last response)
            if args.cos_sim_bert_reward:
                pre_utt = history[-1]
                pre_utt_text = tokenizer.decode(pre_utt, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # get the BERT-based embeddings
                pre_utt_vec = bert_vector(pre_utt_text, bert_emb_tokenizer, bert_emb_model, args)
                response_vec = bert_vector(response, bert_emb_tokenizer, bert_emb_model, args)
                # calculate cosine similarity
                cos_sim_bert_score = cosine_similarity(pre_utt_vec.reshape(1, -1), response_vec.reshape(1, -1))[0][0]
                current_cos_sim_bert_reward = cos_sim_bert_score
                logger.info('cos_sim_bert = %f', current_cos_sim_bert_reward)
                cos_sim_bert_reward = current_cos_sim_bert_reward

            # internal repetition
            if args.intern_rep_reward:
                # intrep_word
                response_tok = response_org.split()
                intrep_1gram = intrep_frac(response_tok)
                # intrep_2gram
                response_tok_2gram = get_ngrams(response, 2)
                intrep_2gram = intrep_frac(response_tok_2gram)
                # intrep_3gram
                response_tok_3gram = get_ngrams(response, 3)
                intrep_3gram = intrep_frac(response_tok_3gram)
                current_intern_rep_reward = 1 - intrep_1gram   # TODO: How to design this reward?
                logger.info('intern_rep = %f', current_intern_rep_reward)
                intern_rep_reward = current_intern_rep_reward

            # external repetition
            if args.extern_rep_reward:
                pre_responses = []
                for j in range(-len(history), 0):
                    if j % 2 == 0:
                        current_text = tokenizer.decode(history[j], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        pre_responses.append(current_text)

                # extrep_word
                response_tok = response.split()   # FIXME: response_org
                prev_tok = [s.split() for s in pre_responses]  # list of list of ints
                prev_tok = list(set(flatten(prev_tok)))  # list of ints, no duplicates
                extrep_1gram = extrep_frac(response_tok, prev_tok)
                # extrep_2gram
                response_tok_2gram = get_ngrams(response, 2)
                prev_2grams = [get_ngrams(prev, 2) for prev in pre_responses]  # list of list of strings
                prev_2grams = list(set(flatten(prev_2grams)))  # list of strings, no duplicates
                extrep_2gram = extrep_frac(response_tok_2gram, prev_2grams)
                # extrep_3gram
                response_tok_3gram = get_ngrams(response, 3)
                prev_3grams = [get_ngrams(prev, 3) for prev in pre_responses]  # list of list of strings
                prev_3grams = list(set(flatten(prev_3grams)))  # list of strings, no duplicates
                extrep_3gram = extrep_frac(response_tok_3gram, prev_3grams)

                current_extern_rep_reward = 0  # TODO: How to design this reward?
                logger.info('extern_rep = %f', current_extern_rep_reward)
                extern_rep_reward += current_extern_rep_reward

            # fine-tuned GPT-based language model
            if args.lm_reward:
                logger.info('RESPONSE: %s', response)
                lm_tokenize_input = lm_tokenizer.tokenize(response)
                lm_tensor_input = torch.tensor([[special_tokens_ids[0]] + lm_tokenizer.convert_tokens_to_ids(lm_tokenize_input) + [special_tokens_ids[-1]]]).to(args.device)
                lm_loss = lm_model(lm_tensor_input, lm_labels=lm_tensor_input)
                nll = - lm_loss.item()
                # threshold -4 --> around PPL 50
                if nll < -4:
                    nll = -4
                current_lm_score = (nll + 4) / 4   # scaled value
                current_lm_reward = current_lm_score
                logger.info('lm_reward = %f', current_lm_reward)
                lm_reward = current_lm_reward

            # question-back
            if args.qback_reward:
                response_tok = response_org.split()
                num_in_list = len([w for w in response_tok if w in QN_WORDS])
                current_qback_reward = num_in_list / len(response_tok)
                logger.info('qback_reward = %f', current_qback_reward)
                qback_reward = current_qback_reward

            # other itmes using labels:
            if args.f1_reward or args.bleu_reward:
                label_text = tokenizer.decode(datasets['train']['labels'][sample_index], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                logger.info("label_text: %s", label_text)
                if args.f1_reward:   # F1 score
                    f1 = f1_rewarder(response, label_text)
                    current_f1_reward = f1 * 1
                    logger.info("f1 = %f", current_f1_reward)
                    f1_reward += current_f1_reward
                if args.bleu_reward:   # BLEU
                    bleu = bleu_rewarder(response, label_text)
                    current_bleu_reward = bleu * 1
                    logger.info('bleu = %f', current_bleu_reward)
                    bleu_reward += current_bleu_reward

            # the final reward
            current_reward = persona_reward * args.nli_weight + response_reward * args.nli_allres_weight + \
                             cos_sim_bert_reward * args.cos_sim_bert_weight + intern_rep_reward * args.intern_rep_weight + \
                             lm_reward * args.lm_weight + qback_reward * args.qback_weight
            logger.info('reward = %f', current_reward)
            rewards.append(current_reward)
            if args.nli_reward:
                persona_rewards.append(persona_reward)
            if args.nli_allres_reward:
                response_rewards.append(response_reward)
            if args.cos_sim_bert_reward:
                cos_sim_bert_rewards.append(cos_sim_bert_reward)
            if args.intern_rep_reward:
                intern_rep_rewards.append(intern_rep_reward)
            if args.extern_rep_reward:
                extern_rep_rewards.append(extern_rep_reward)
            if args.lm_reward:
                lm_rewards.append(lm_reward)
            if args.qback_reward:
                qback_rewards.append(qback_reward)
            if args.f1_reward:
                f1_rewards.append(f1_reward)
            if args.bleu_reward:
                bleu_rewards.append(bleu_reward)

            # Change the sample as label for reinforcement learning:
            for k, v in itertools.groupby(lm_labels[i][-1]):
                if k == -1:
                    len_left_pad = len(list(v))
                    break
            len_right_pad = len(lm_labels[i][-1]) - len_left_pad - len(response_ids)

            mc_token_ids[i][-1] = len_left_pad + len(response_ids)

            # revise the array length after changing the generated sample as label
            if len_right_pad > 0:
                input_ids[i][-1][len_left_pad: len_left_pad + len(response_ids)] = response_ids.copy()
                input_ids[i][-1][len_left_pad + len(response_ids):] = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])] * len_right_pad

                lm_labels[i][-1] = [-1] * len_left_pad + response_ids + [-1] * len_right_pad

                token_type_ids[i][-1][len_left_pad: len_left_pad + len(response_ids)] = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[2])] * len(response_ids)
                token_type_ids[i][-1][len_left_pad + len(response_ids):] = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])] * len_right_pad
            else:
                # print('ERROR DETECTION - len_right_pad <<<<<<<<<<<<<<<<<<=================== 0!!!!!!!!!!!!!!!!!!!!!!!')
                # print('ERROR DETETION - len_right_pad=', len_right_pad)
                input_ids[i][-1][len_left_pad:] = response_ids[: (len(lm_labels[i][-1]) - len_left_pad)]

                mc_token_ids[i][-1] = len(lm_labels[i][-1]) - 1

                lm_labels[i][-1] = [-1] * len_left_pad + response_ids[: (len(lm_labels[i][-1]) - len_left_pad)]

                token_type_ids[i][-1][len_left_pad:] = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[2])] * ((len(lm_labels[i][-1]) - len_left_pad))   # + len(response_ids)
            batch_size += 1

            # print('------------------------------------')

        # Sample as label:
        input_ids = Variable(torch.LongTensor(np.array(input_ids)).contiguous()).to(args.device)
        mc_token_ids = Variable(torch.LongTensor(np.array(mc_token_ids)).contiguous()).to(args.device)
        lm_labels = Variable(torch.LongTensor(np.array(lm_labels)).contiguous()).to(args.device)
        token_type_ids = Variable(torch.LongTensor(np.array(token_type_ids)).contiguous()).to(args.device)
        batch = (input_ids, mc_token_ids, lm_labels, batch[3], token_type_ids, batch[5])

        reward = sum(rewards) / batch_size   # average reward in the current batch
        train_metrics['reward'].append(reward)
        # add the rewards to train_matrics for plotting
        if args.nli_reward:
            train_metrics['persona_reward'].append(sum(persona_rewards) / batch_size)
        if args.nli_allres_reward:
            train_metrics['response_reward'].append(sum(response_rewards) / batch_size)
        if args.cos_sim_bert_reward:
            train_metrics['cos_sim_bert_reward'].append(sum(cos_sim_bert_rewards) / batch_size)
        if args.intern_rep_reward:
            train_metrics['intern_rep_reward'].append(sum(intern_rep_rewards) / batch_size)
        if args.extern_rep_reward:
            train_metrics['extern_rep_reward'].append(sum(extern_rep_rewards) / batch_size)
        if args.lm_reward:
            train_metrics['lm_reward'].append(sum(lm_rewards) / batch_size)
        if args.qback_reward:
            train_metrics['qback_reward'].append(sum(qback_rewards) / batch_size)
        if args.f1_reward:
            train_metrics['f1_reward'].append(sum(f1_rewards) / batch_size)
        if args.bleu_reward:
            train_metrics['bleu_reward'].append(sum(bleu_rewards) / batch_size)

        model.eval()
        model.lm_head.train()
        # lock other layers in the model except the top layer of the generative part
        for p in model.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = True

        # forward_start = time.time()   # used to calculate the execution time for getting the loss through the model
        losses, shift_logits = model(*batch[:-1], rl_train=True)
        # forward_end = time.time()
        # logger.info("Forward execution time = %f", forward_end-forward_start)

        lm_loss, mc_loss = losses

        rewards_var = Variable(torch.FloatTensor(np.array(rewards)).contiguous()).to(args.device)
        rewards_var = rewards_var.unsqueeze(0).t().expand(rewards_var.shape[0], input_ids.shape[-1])[:, :-1]
        rewards_var = rewards_var.unsqueeze(1).expand(rewards_var.shape[0], input_ids.shape[-2], rewards_var.shape[-1])

        # lm_logits = torch.FloatTensor(shift_logits.cpu().detach().numpy()).to(args.device)
        lm_logits = Variable(shift_logits.detach())

        for i in range(args.critic_train):
            baselines = critic_model(lm_logits).squeeze(3)   # get the baseline value from the critic model
            critic_loss = critic_criterion(baselines, rewards_var)   # get the critic loss

            norm_rewards = (rewards_var.detach() - baselines.detach()).view(-1)   # normalize the critic reward

            # update the critic model
            critic_loss.backward()
            critic_optimizer.step()
            critic_optimizer.zero_grad()

        lm_loss = torch.mean(lm_loss.mul(norm_rewards))
        loss = lm_loss / args.gradient_accumulation_steps
        # RL training finished

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    rl_trainer = Engine(rl_update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, sample_index = batch
            logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
            lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)


    # Attach evaluation to rl_trainer: we evaluate when we start the training and at the end of each epoch
    rl_trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        rl_trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        rl_trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        rl_trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    rl_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(rl_trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])

    for name, metric in metrics.items():
        metric.attach(evaluator, name)


    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(rl_trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))


        tb_logger = TensorboardLogger(log_dir=None)

        tb_logger.attach(rl_trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(rl_trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=rl_trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        rl_trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    # YW: RL part:
    # Run the training
    # from resp_generator import GeneratorAgent
    # generator_actor = GeneratorAgent(args=args, model=model)

    # Load the fine-tuned GPT-based language model
    if args.lm_reward:
        reset_seed(args.seed)   # IMPORTANT!!! Do the seed resetting each time before defining or loading a model!!!
        lm_model_path = 'openai-gpt'
        lm_output_dir = 'language-quality-subreward/gpt_output'
        lm_special_tokens = ['_start_', '_delimiter_', '_classify_']
        # Load pre-trained model (weights)
        with torch.no_grad():
            lm_output_config_file = os.path.join(lm_output_dir, CONFIG_NAME)
            lm_config = OpenAIGPTConfig(lm_output_config_file)

            lm_output_model_file = os.path.join(lm_output_dir, WEIGHTS_NAME)
            # lm_model_state_dict = torch.load(lm_output_model_file, map_location='cpu')
            lm_model_state_dict = torch.load(lm_output_model_file)
            lm_model = OpenAIGPTLMHeadModel(lm_config)
            lm_model.load_state_dict(lm_model_state_dict)

            # Load pre-trained model tokenizer (vocabulary)
            lm_tokenizer = OpenAIGPTTokenizer.from_pretrained(lm_model_path,
                                                                cache_dir='./tmp/',
                                                               special_tokens=lm_special_tokens)

        special_tokens_ids = list(lm_tokenizer.convert_tokens_to_ids(token) for token in lm_special_tokens)
        lm_model.to(args.device)
        lm_model.eval()

    # BERT embedding
    if args.cos_sim_bert_reward:
        reset_seed(args.seed)
        bert_emb_modelpath = "bert-base-uncased"
        bert_emb_tokenizer = BertTokenizer.from_pretrained(bert_emb_modelpath,cache_dir='./tmp/')
        bert_emb_model = BertModel.from_pretrained(bert_emb_modelpath,cache_dir='./tmp/').to(args.device)
        bert_emb_model.eval()

    # pu (persona-utterance) NLI
    if args.nli_reward or args.nli_allres_reward:
        reset_seed(args.seed)
        nli_tokenizer = BertTokenizer.from_pretrained(args.bert_model,cache_dir='./tmp/', do_lower_case=args.do_lower_case)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

        nli_config = BertConfig(output_config_file)
        nli_model = BertForSequenceClassification(nli_config, num_labels=3)
        nli_model.load_state_dict(torch.load(output_model_file))
        nli_model.to(args.device)
        nli_model.eval()

    # Critic model - Linear layer
    reset_seed(args.seed)
    critic_model = LinearRegressionModel()
    critic_model.to(args.device)
    critic_criterion = torch.nn.MSELoss(size_average=False)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=0.0001)
    critic_model.train()

    train_metrics = defaultdict(list)

    try:
        rl_trainer.run(train_loader, max_epochs=args.n_epochs)
    except KeyboardInterrupt:
        print("Training stopped by keyboard.")
    # plotting
    if args.nli_reward:
        plot_reward(train_metrics, 'persona_reward', 'results/'+tb_logger.writer.log_dir[5:], 'persona_reward' + '.jpg', interval=args.plot_interval)
    if args.nli_allres_reward:
        plot_reward(train_metrics, 'response_reward', 'results/' + tb_logger.writer.log_dir[5:], 'response_reward' + '.jpg', interval=args.plot_interval)
    if args.cos_sim_bert_reward:
        plot_reward(train_metrics, 'cos_sim_bert_reward', 'results/' + tb_logger.writer.log_dir[5:], 'cos_sim_bert_reward' + '.jpg', interval=args.plot_interval)
    if args.intern_rep_reward:
        plot_reward(train_metrics, 'intern_rep_reward', 'results/' + tb_logger.writer.log_dir[5:], 'intern_rep_reward' + '.jpg', interval=args.plot_interval)
    if args.extern_rep_reward:
        plot_reward(train_metrics, 'extern_rep_reward', 'results/' + tb_logger.writer.log_dir[5:], 'extern_rep_reward' + '.jpg', interval=args.plot_interval)
    if args.lm_reward:
        plot_reward(train_metrics, 'lm_reward', 'results/' + tb_logger.writer.log_dir[5:], 'lm_reward' + '.jpg', interval=args.plot_interval)
    if args.qback_reward:
        plot_reward(train_metrics, 'qback_reward', 'results/' + tb_logger.writer.log_dir[5:], 'qback_reward' + '.jpg', interval=args.plot_interval)
    if args.f1_reward:
        plot_reward(train_metrics, 'f1_reward', 'results/'+tb_logger.writer.log_dir[5:], 'f1_reward' + '.jpg', interval=args.plot_interval)
    if args.bleu_reward:
        plot_reward(train_metrics, 'bleu_reward', 'results/'+tb_logger.writer.log_dir[5:], 'bleu_reward' + '.jpg', interval=args.plot_interval)
    plot_reward(train_metrics, 'reward', 'results/' + tb_logger.writer.log_dir[5:], 'all_reward' + '.jpg', interval=args.plot_interval)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    try:
        if args.local_rank in [-1, 0] and args.n_epochs > 0:
            os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir,
                                                                         WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
            tb_logger.close()
    except Exception as e:
        traceback.print_exc(e)


if __name__ == "__main__":
    train()
