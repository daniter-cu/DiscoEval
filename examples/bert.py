# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import torch
import code
import argparse

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import discoeval 

from transformers import BertConfig, BertTokenizer, BertModel, BertForPreTraining

# SentEval prepare and batcher
def prepare(params, samples):
    pass


def batcher_bak(params, batch):
    layer = params["layer"]
    model = params["model"]
    tokenizer = params.tokenizer
    batch = [[token.lower() for token in sent] for sent in batch]
    batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    batch = [["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"] for sent in batch]
    batch = [b[:512] for b in batch]
    seq_length = max([len(sent) for sent in batch])
    mask = [[1] * len(sent) + [0] * (seq_length - len(sent)) for sent in batch]
    segment_ids = [[0] * seq_length for _ in batch]
    batch = [tokenizer.convert_tokens_to_ids(sent) + [0] * (seq_length - len(sent)) for sent in batch]
    with torch.no_grad():
        batch = torch.tensor(batch).cuda()
        mask = torch.tensor(mask).cuda()  # bs * seq_length
        segment_ids = torch.tensor(segment_ids).cuda()
        outputs, pooled_output, hidden_states, _ = model(batch, token_type_ids=segment_ids, attention_mask=mask)
        if layer == "avg":
            output = [o.data.cpu()[:, 0].numpy() for o in hidden_states]
            embeddings = np.mean(output, 0)
        elif layer == "pooler":
            embeddings = pooled_output.data.cpu().numpy()
        else:
            layer = int(layer)
            output = hidden_states[layer]
            embeddings = output.data.cpu()[:, 0].numpy()

    return embeddings


def batcher(params, batch, batch2=None):
    layer = params["layer"]
    model = params["model"]
    tokenizer = params.tokenizer
    batch = [[token.lower() for token in sent] for sent in batch]
    batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    batch = [["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"] for sent in batch]
    segment_1 = [[0] * len(seq) for seq in batch]
    if batch2 is not None:
        batch2 = [[token.lower() for token in sent] for sent in batch2]
        batch2 = [" ".join(sent) if sent != [] else "." for sent in batch2]
        batch2 = [tokenizer.tokenize(sent) + ["[SEP]"] for sent in batch2]
        for i in range(len(batch)):
            if len(batch[i]) + len(batch2[i]) > 512:
                while len(batch[i]) + len(batch2[i]) > 512:
                    if len(batch[i]) > len(batch2[i]):
                        del batch[i][-2]
                    else:
                        del batch2[i][-2]
        segment_1 = [[0] * len(seq) for seq in batch]
        segment_2 = [[1] * len(seq) for seq in batch2]
        batch = [a+b for a,b in zip(batch, batch2)]
    else:
        segment_2 = [[] for _ in batch]
        segment_1 = [sg[:512] for sg in segment_1]

    batch = [b[:512] for b in batch]
    seq_length = max([len(sent) for sent in batch])
    mask = [[1]*len(sent) + [0]*(seq_length-len(sent)) for sent in batch]
    # segment_ids = [[0]*seq_length for _ in batch]
    segment_ids = [a + b for a,b in zip(segment_1, segment_2)]
    segment_ids = [si + [0] * (seq_length - len(si)) for si in segment_ids]
    batch = [tokenizer.convert_tokens_to_ids(sent) + [0]*(seq_length - len(sent)) for sent in batch]
    with torch.no_grad():
        batch = torch.tensor(batch).cuda()
        mask = torch.tensor(mask).cuda() # bs * seq_length
        segment_ids = torch.tensor(segment_ids).cuda()   	
        outputs, pooled_output, hidden_states, _ = model(batch, token_type_ids=segment_ids, attention_mask=mask)
        if layer == "avg":
            output = [o.data.cpu()[:, 0].numpy() for o in hidden_states]
            embeddings = np.mean(output, 0)
        elif layer == "pooler":
            embeddings = pooled_output.data.cpu().numpy()
        elif layer == "custom":
            embeddings = [np.mean(o.data.cpy().numpy(), axis=0) for o in outputs]
        else: 
            layer = int(layer)
            output = hidden_states[layer]
            embeddings = output.data.cpu()[:, 0].numpy()
 
    return embeddings



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_index", default=0, type=int, required=True,
                        help="which task to perform")
    parser.add_argument("--layer", default="avg", type=str, required=True,
                        help="which layer to evaluate on")
    parser.add_argument("--model_type", default="base", type=str, required=True, choices=["base", "large"],
                        help="the type of BERT model to evaluate on")
    parser.add_argument("--weights", default="conpono", type=str, required=True, choices=["bert", "conpono"],
                        help="model weights")
    parser.add_argument("--data_path", default="./data/", type=str, required=False,
                        help="data path")
    parser.add_argument("--cache_path", default="/tmp/", type=str, required=False,
                        help="cache path")
    parser.add_argument("--deepnet", default="False", type=bool, required=False,
                        help="unused")
    parser.add_argument("--kval", default=1, type=int, required=False,
                        help="for ablations")
    parser.add_argument("--seed", default=111, type=int, required=False,
                        help="random seed")
    args = parser.parse_args()

    # Set up logger
    PATH_TO_DATA = args.data_path
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    config = BertConfig.from_pretrained('bert-{}-uncased'.format(args.model_type))
    config.output_hidden_states = True
    config.output_attentions = True
    tokenizer = BertTokenizer.from_pretrained('bert-{}-uncased'.format(args.model_type))
    if args.weights == "bert":
        model = BertModel.from_pretrained('bert-{}-uncased'.format(args.model_type), cache_dir=args.cache_path, config=config).cuda()
    elif args.weights == "conpono":
        model = BertModel.from_pretrained('./models/conpono', config=config).cuda()
    model.eval()

    # Set params for SentEval
    params_discoeval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 8,
                        'tokenizer': tokenizer, "layer": args.layer, "model": model, "seed": args.seed}
    params_discoeval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                     'tenacity': 5, 'epoch_size': 4}
    se = discoeval.engine.SE(params_discoeval, batcher, prepare)


    transfer_tasks = [
        ['SParxiv', 'SProc', 'SPwiki'], 
        ['DCchat', 'DCwiki'],
        ['BSOarxiv', 'BSOroc', 'BSOwiki'], 
        ['SSPabs', 'RST'],
        [ 'PDTB-E', 'PDTB-I']]
    results = se.eval(transfer_tasks[args.task_index])
    print(results)
