# Reference: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py

import re
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.idx_to_str = ['<PAD>', '<s>', '</s>', '<UNK>']
        self.str_to_idx = {s: idx for idx, s in enumerate(self.idx_to_str)}
        self.start_idx = len(self.idx_to_str) # length at which real tokens starts

    def __len__(self):
        return len(self.idx_to_str)

    @staticmethod
    def tokenize(line):
        # TODO: try better tokenizers
        return re.findall(r'\w+', line.lower())

    def build_vocab(self, lines):
        # TODO: use GloVe
        frequencies = defaultdict(int)
        curr_idx = self.start_idx
        for line in lines:
            tokens = self.tokenize(line)
            for token in tokens:
                frequencies[token] += 1
                if frequencies[token] == self.freq_threshold:
                    self.idx_to_str.append(token)
                    self.str_to_idx[token] = curr_idx
                    curr_idx += 1

    def numericalize(self, line):
        """
        Call this only after the vocab has been built
        """
        tokens = self.tokenize(line)
        ret = [self.str_to_idx['<s>']]
        for token in tokens:
            if token in self.str_to_idx:
                ret.append(self.str_to_idx[token])
            else:
                ret.append(self.str_to_idx['<UNK>'])
        ret.append(self.str_to_idx['</s>'])
        return ret

    def denumericalize(self, token_indices):
        """
        Invert numericalize, returns a string
        """
        # remove start and end token
        ret = []
        for idx in token_indices[1 : -1]:
            token = self.idx_to_str[idx]
            # break early when hitting <PAD> token
            if token == '<PAD>':
                break
            else:
                ret.append(token)
        return ' '.join(ret)

class TrainDataset(Dataset):
    def __init__(self, filename, freq_threshold=5, num_transforms=3):
        """
        num_transforms: number of transforms to apply to the line to generate a negative sample
        """
        self.num_transforms = num_transforms
        self.first_column_lines = [] # lines in the first column
        self.second_column_lines = []
        with open(filename, 'rt') as f:
            for line in f:
                # do minimal amount of preprocessing here, lowercasing is done in vocab
                first, second = line.split('\t')
                self.first_column_lines.append(first)
                self.second_column_lines.append(second)

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.first_column_lines + self.second_column_lines)

    def __len__(self):
        return len(self.first_column_lines)

    def generate_negative_example(self, numericalized_line):
        # randomly substitute in words after vocab.start_idx
        # TODO: insertion, deletion, permutation
        ret = numericalized_line.copy()
        # position in line to perturb
        token_indices = np.random.choice(range(len(numericalized_line)),
        self.num_transforms, replace=False)
        vocab_indices = np.random.choice(range(self.vocab.start_idx,
        len(self.vocab)), self.num_transforms)
        for tok_idx, vocab_idx in zip(token_indices, vocab_indices):
            ret[tok_idx] = vocab_idx
        return ret

    def __getitem__(self, index):
        """
        Return a triplet of numericalized lines
        original, positive, and negative example
        (line, paraphrased line, non-paraphrasal line)
        """
        line = self.first_column_lines[index]
        positive_line = self.second_column_lines[index]
        # convert tokens to indices
        numericalized_line = self.vocab.numericalize(line)
        numericalized_positive = self.vocab.numericalize(positive_line)
        # generate a negative numericalized example
        numericalized_negattive = self.generate_negative_example(numericalized_line)

        return numericalized_line, numericalized_positive, numericalized_negattive
