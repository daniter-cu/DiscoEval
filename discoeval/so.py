# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Sentence Position (SP), Binary Sentence Ordering (BSO), Discourse Coherence (DC)
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np
import copy
from tqdm import tqdm

from discoeval.tools.validation import SplitClassifier

class SPEval(object):
    """Sentence Position Evaluation
       Given five sentences, predict the location of the first sentence. 
    """
    def __init__(self, task_path, task_name, nclasses=5, seed=1111):
        self.seed = seed
        np.random.seed(seed)

        self.nclasses = nclasses
        self.task_name = task_name
        logging.debug('***** Transfer task : Sentence Ordering task. Task name: {} *****\n\n'.format(self.task_name))

        train = self.loadFile(os.path.join(task_path, 'train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'valid.txt'))
        test = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.data = {'train': train, 'valid': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = [sent for sents in self.data['train']["X"] for sent in sents] + \
                    [sent for sents in self.data['valid']["X"] for sent in sents] + \
                    [sent for sents in self.data['test']["X"] for sent in sents]
        return prepare(params, samples)

    def loadFile(self, fpath):
        data = {"X": [], "y": []}
        buf = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split("\t")
                data["X"].append([sent.split() for sent in line[1:]])
                data["y"].append(int(line[0]))
        logging.debug('Loaded {} instances\n'.format(len(data["X"])))
        return data

    def run(self, params, batcher):
        embed = {'train': {}, 'valid': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            x_data_filename = "/tmp/SP-%s-conpono-%s-x.npy" % (self.task_name, key)
            y_data_filename = "/tmp/SP-%s-conpono-%s-y.npy" % (self.task_name, key)
            if os.path.isfile(x_data_filename):
                assert os.path.isfile(y_data_filename), "Labels don't exist"
                embed[key]['X'] = np.load(x_data_filename)
                embed[key]['y'] = np.load(y_data_filename)
            else:
                logging.info('Computing embedding for {0}'.format(key))
                embed[key]['X'] = []
                for j in tqdm(range(0, len(self.data[key]['X']), bsize), total=len(self.data[key['X']]) / bsize):
                    batch_embed = []
                    batch1 = [sents[0] for sents in self.data[key]['X'][j:j + bsize]]
                    embed1 = batcher(params, batch1)
                    batch_embed.append(embed1)
                    for i in range(1,5):
                        batch2 = [sents[i] for sents in self.data[key]['X'][j:j + bsize]]
                        embed2 = batcher(params, batch1, batch2)
                        batch_embed.append(embed2)
                    embed[key]['X'].append(np.hstack(batch_embed))
                embed[key]['X'] = np.vstack(embed[key]['X'])
                embed[key]['y'] = np.array(self.data[key]['y'])
                np.save(x_data_filename, embed[key]['X'])
                np.save(y_data_filename, embed[key]['y'])
                logging.info('Computed {0} embeddings, shape: {1}'.format(key, embed[key]["X"].shape))



        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'noreg': True}
        params.classifier['nhid'] = 50
        params.classifier['tenacity'] = 20
        params.classifier['epoch_size'] = 1
        params.classifier['max_epoch'] = 20

        clf = SplitClassifier(X={'train': embed['train']['X'],
                                 'valid': embed['valid']['X'],
                                 'test': embed['test']['X']},
                              y={'train': embed['train']['y'],
                                 'valid': embed['valid']['y'],
                                 'test': embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nTask: {0}, Dev acc : {1} Test acc : {2} for \
            Sentence position prediction\n'.format(self.task_name, devacc, testacc))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(embed['valid']['X']),
                'ntest': len(embed['test']['X'])}



class BSOEval(object):
    """Binary Sentence Ordering Evaluation
    """
    def __init__(self, task_path, task_name, nclasses=5, seed=1111):
        self.seed = seed
        np.random.seed(seed)

        self.nclasses = nclasses
        self.task_name = task_name
        logging.debug('***** Transfer task : Binary Sentece Ordering. Task name: {} *****\n\n'.format(self.task_name))

        self.data = {}
        data = self.loadFile(os.path.join(task_path, 'train.txt'))
        self.data["train"] = {"X": data}
        data = self.loadFile(os.path.join(task_path, 'valid.txt'))
        self.data["valid"] = {"X": data}
        data = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.data["test"] = {"X": data} 

    def do_prepare(self, params, prepare):
        samples = [sent for sents in self.data['train']["X"] for sent in sents] + \
                    [sent for sents in self.data['valid']["X"] for sent in sents] + \
                    [sent for sents in self.data['test']["X"] for sent in sents]
        return prepare(params, samples)

    def loadFile(self, fpath):
        data = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append([sent.split() for sent in line.strip().split("\t")])
        logging.debug('Loaded {} instances\n'.format(len(data)))
        return data

    def run(self, params, batcher):
        embed = {'train': {}, 'valid': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            pos_data_filename = "/tmp/BSO-%s-conpono-%s-x.npy" % (self.task_name, key)
            neg_data_filename = "/tmp/BSO-%s-conpono-%s-y.npy" % (self.task_name, key)
            if os.path.isfile(pos_data_filename):
                assert os.path.isfile(neg_data_filename), "Labels don't exist"
                pos = np.load(pos_data_filename)
                neg = np.load(neg_data_filename)
            else:
                logging.info('Computing embedding for {0}'.format(key))
                pos, neg = [], []
                for ii in tqdm(range(0, len(self.data[key]['X']), bsize), total=len(self.data[key]['X'])/ bsize):
                    batch1 = [sents[0] for sents in self.data[key]['X'][ii:ii + bsize]]
                    batch2 = [sents[1] for sents in self.data[key]['X'][ii:ii + bsize]]
                    embeddings = batcher(params, batch1, batch2)
                    neg_embeddings = batcher(params, batch2, batch1)
                    pos.append(embeddings)
                    neg.append(neg_embeddings)
                pos = np.vstack(pos)
                neg = np.vstack(neg)
                np.save(pos_data_filename, pos)
                np.save(neg_data_filename, neg)

            y = [1] * len(pos) + [0] * len(neg)
            permute = np.random.permutation(len(y))
            X = np.concatenate([pos, neg], 0)
            X = np.concatenate([copy.deepcopy(X[i][None, :]) for i in permute], 0)
            y = np.array([y[i] for i in permute])
            embed[key]["X"] = X
            embed[key]["y"] = y


            logging.info('Computed {0} embeddings, shape: {1}'.format(key, embed[key]["X"].shape))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'noreg': True}

        clf = SplitClassifier(X={'train': embed['train']['X'],
                                 'valid': embed['valid']['X'],
                                 'test': embed['test']['X']},
                              y={'train': embed['train']['y'],
                                 'valid': embed['valid']['y'],
                                 'test': embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nTask: {0}, Dev acc : {1} Test acc : {2} for \
            Binary Sentence Ordering prediction\n'.format(self.task_name, devacc, testacc))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(embed['valid']['X']),
                'ntest': len(embed['test']['X'])}


class DCEval(object):
    """Discourse Coherence Evaluation
    """
    def __init__(self, task_path, task_name, nclasses=2, seed=1111):
        self.seed = seed
        self.task_name = task_name

        assert nclasses in [2]
        self.nclasses = nclasses
        logging.debug('***** Transfer task : Discourse Coherence classification, task name: {} *****\n\n'.format(self.task_name))

        train = self.loadFile(os.path.join(task_path, 'train.txt'))
        valid = self.loadFile(os.path.join(task_path, 'valid.txt'))
        test = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.data = {'train': train, 'valid': valid, 'test': test}

    def do_prepare(self, params, prepare):
        samples = [sent for sents in self.data['train']['X'] for sent in sents] + [sent for sents in self.data['valid']['X']  for sent in sents] + [sent for sents in self.data['test']['X']  for sent in sents]
        return prepare(params, samples)

    def loadFile(self, fpath):
        data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split("\t")
                data["X"].append([sent.split() for sent in line[1:]])
                assert line[0] in ["0", "1"], "undefined category, a category has to be either 0 or 1"
                data["y"].append(int(line[0]))
        logging.debug('Loaded {} instances\n'.format(len(data["y"])))
        return data

    def run(self, params, batcher):
        embed = {'train': {}, 'valid': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            x_data_filename = "/tmp/DC-%s-conpono-%s-x.npy" % (self.task_name, key)
            y_data_filename = "/tmp/DC-%s-conpono-%s-y.npy" % (self.task_name, key)
            if os.path.isfile(x_data_filename):
                assert os.path.isfile(y_data_filename), "Labels don't exist"
                embed[key]['X'] = np.load(x_data_filename)
                embed[key]['y'] = np.load(y_data_filename)
            else:
                logging.info('Computing embedding for {0}'.format(key))
                embeddings_i = []
                for i in tqdm(range(0, len(self.data[key]['X']), bsize), total=len(self.data[key]['X'])/bsize):
                    para_embeddings = []
                    for j in range(0,6,2):
                        batch1 = [sents[j:j+1] for sents in self.data[key]['X'][i:i+bsize]]
                        batch2 = [sents[j+1:j+2] for sents in self.data[key]['X'][i:i+bsize]]
                        embeddings = batcher(params, batch1, batch2)
                        para_embeddings.append(embeddings)
                    embeddings_i.append(np.hstack(para_embeddings))
                embed[key]['X'] = np.vstack(embeddings_i)
                embed[key]['y'] = np.array(self.data[key]['y'])
                np.save(x_data_filename, embed[key]['X'])
                np.save(y_data_filename, embed[key]['y'])
                logging.info('Computed {0} embeddings, shape: {1}'.format(key, embed[key]["X"].shape))

        params.classifier["nhid"] = 2000
        params.classifier["tenacity"] = 80
        params.classifier["epoch_size"] = 4
        params.classifier["max_epoch"] = 40
        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'noreg': True, 
                             'classifier': params.classifier}

        clf = SplitClassifier(X={'train': embed['train']['X'],
                                 'valid': embed['valid']['X'],
                                 'test': embed['test']['X']},
                              y={'train': embed['train']['y'],
                                 'valid': embed['valid']['y'],
                                 'test': embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            Discourse Coherence classification task\n'.format(devacc, testacc))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(embed['valid']['X']),
                'ntest': len(embed['test']['X'])}

class SSPEval(object):
    """Sentence Section Prediction Evaluation
    """
    def __init__(self, task_path, task_name, nclasses=2, seed=1111):
        self.seed = seed
        self.task_name = task_name

        assert nclasses in [2]
        self.nclasses = nclasses
        logging.debug('***** Transfer task : Sentence Section Prediction, task name: {} *****\n\n'.format(self.task_name))

        train = self.loadFile(os.path.join(task_path, 'train.txt'))
        valid = self.loadFile(os.path.join(task_path, 'valid.txt'))
        test = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.data = {'train': train, 'valid': valid, 'test': test}

    def do_prepare(self, params, prepare):
        samples = [sent for sents in self.data['train']['X'] for sent in sents] + [sent for sents in self.data['valid']['X']  for sent in sents] + [sent for sents in self.data['test']['X']  for sent in sents]
        return prepare(params, samples)

    def loadFile(self, fpath):
        data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split("\t")
                data["X"].append(line[1].split())
                assert line[0] in ["0", "1"], "undefined category, a category has to be either 0 or 1"
                data["y"].append(int(line[0]))
        logging.debug('Loaded {} instances\n'.format(len(data["y"])))
        return data

    def run(self, params, batcher):
        embed = {'train': {}, 'valid': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            x_data_filename = "/tmp/SSP-%s-conpono-%s-x.npy" % (self.task_name, key)
            y_data_filename = "/tmp/SSP-%s-conpono-%s-y.npy" % (self.task_name, key)
            if os.path.isfile(x_data_filename):
                assert os.path.isfile(y_data_filename), "Labels don't exist"
                embed[key]['X'] = np.load(x_data_filename)
                embed[key]['y'] = np.load(y_data_filename)
            else:
                logging.info('Computing embedding for {0}'.format(key))
                embed[key]['X'] = []
                embeddings = []
                for ii in range(0, len(self.data[key]["X"]), bsize):
                    batch = self.data[key]["X"][ii:ii + bsize]
                    embeddings.append(batcher(params, batch))
                embeddings = np.vstack(embeddings)
                embed[key]["X"] = embeddings
                embed[key]['y'] = np.array(self.data[key]['y'])
                np.save(x_data_filename, embed[key]['X'])
                np.save(y_data_filename, embed[key]['y'])
                logging.info('Computed {0} embeddings, shape: {1}'.format(key, embed[key]["X"].shape))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'noreg': True, 
                             'classifier': params.classifier}

        clf = SplitClassifier(X={'train': embed['train']['X'],
                                 'valid': embed['valid']['X'],
                                 'test': embed['test']['X']},
                              y={'train': embed['train']['y'],
                                 'valid': embed['valid']['y'],
                                 'test': embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            Discourse Coherence classification task\n'.format(devacc, testacc))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(embed['valid']['X']),
                'ntest': len(embed['test']['X'])}

