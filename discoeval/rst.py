# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
RST-DT - Task
'''
from __future__ import absolute_import, division, unicode_literals

import pickle
import os
import io
import copy
import logging
import numpy as np

from discoeval.tools.validation import SplitClassifier


class RSTEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : RST-DT Task at {}*****\n\n'.format(taskpath))
        self.seed = seed
        self.task_name = "RST"

        train_sents, train_labels = self.loadFile(os.path.join(taskpath, 'RST_TRAIN.pkl'))
        valid_sents, valid_labels = self.loadFile(os.path.join(taskpath, 'RST_DEV.pkl'))
        test_sents, test_labels = self.loadFile(os.path.join(taskpath, 'RST_TEST.pkl'))

        logging.debug("#train: {}, #dev: {}, #test: {}".format(len(train_sents), len(valid_sents), len(test_sents)))
        self.labelset = set(train_labels + valid_labels + test_labels)
        self.samples = sum(train_sents, []) + sum(valid_sents, []) + sum(test_sents, [])

        logging.debug('***** Total instances loaded: {}*****'.format(len(train_sents + valid_sents + test_sents)))
        logging.debug('***** Total #label categories: {}****\n\n'.format(len(self.labelset)))
        self.data = {'train': (train_sents, train_labels),
                     'valid': (valid_sents, valid_labels),
                     'test': (test_sents, test_labels)
                     }

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        data = pickle.load(open(fpath, "rb"))
        sents = []
        labels = []
        for d in data:
            sents.append([d[1], d[2]])
            labels.append(d[0])
        return sents, labels

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        dico_label = {k: v for v, k in enumerate(self.labelset)}
        for key in self.data:
            x_data_filename = "/tmp/RST-%s-conpono-%s-x.npy" % (self.task_name, key)
            y_data_filename = "/tmp/RST-%s-conpono-%s-y.npy" % (self.task_name, key)
            if os.path.isfile(x_data_filename):
                assert os.path.isfile(y_data_filename), "Labels don't exist"
                self.X[key] = np.load(x_data_filename)
                self.y[key] = np.load(y_data_filename)
            else:
                if key not in self.X:
                    self.X[key] = []
                if key not in self.y:
                    self.y[key] = []

                input, labels = self.data[key]

                logging.debug('split: {}, #data: {}'.format(key, len(input)))
                enc_input = []
                sent2vec = {}
                for ii in range(0, len(input), params.batch_size):
                    batch = input[ii:ii + params.batch_size]
                    seq1 = [b[0] for b in batch]
                    seq2 = [b[1] for b in batch]
                    emb = batcher(params, seq1, seq2)
                    enc_input.append(emb)
                    if ii % (100*params.batch_size) == 0:
                        logging.info("PROGRESS (encoding): %.2f%%" %
                                     (100 * ii / len(input)))
                self.X[key] = np.vstack(enc_input)
                self.y[key] = np.array([dico_label[y] for y in labels])
                np.save(x_data_filename, self.X[key])
                np.save(y_data_filename, self.y[key])


            logging.info("encoding X to be: {}".format(self.X[key].shape))

        config = {'nclasses': len(dico_label), 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for RST-DT\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
