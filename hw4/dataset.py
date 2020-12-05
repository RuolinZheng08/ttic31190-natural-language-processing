import re
import pickle
import numpy as np
from collections import defaultdict

from nltk.tokenize import word_tokenize

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

UNK = '<unk>'
PAD = '<pad>'
SOS = '<s>' # start of sentence
EOS = '</s>'

class GloveVocabulary:
    def __init__(self, glove_vocab_path, glove_emb_path):
        self.idx_to_str = [PAD, SOS, EOS] # <unk> is in GloVe
        self.start_idx = len(self.idx_to_str) # length at which real tokens starts
        # load glove into self.idx_to_str and self.str_to_idx
        with open(glove_vocab_path, 'rb') as f:
            glove_vocab = pickle.load(f)
        with open(glove_emb_path, 'rb') as f:
            glove_emb = pickle.load(f)
        self.idx_to_str += glove_vocab
        self.str_to_idx = {s: idx for idx, s in enumerate(self.idx_to_str)}

        # TODO: initialize emb for special tokens
        # instead of random vector, use the mean of all glove vectors for special tokens
        glove_emb = torch.tensor(glove_emb)
        mean_vec = glove_emb.mean(dim=0, keepdim=True)
        self.embedding = torch.cat(
            [mean_vec, mean_vec, mean_vec, glove_emb], dim=0
        )

    def __len__(self):
        return len(self.idx_to_str)

    @staticmethod
    def tokenize(line):
        # TODO: try different tokenizers
        return word_tokenize(line.lower())

    def numericalize(self, line):
        """
        Call this only after the vocab has been built
        """
        tokens = self.tokenize(line)
        ret = [self.str_to_idx[SOS]]
        for token in tokens:
            if token in self.str_to_idx:
                ret.append(self.str_to_idx[token])
            else:
                ret.append(self.str_to_idx[UNK])
        ret.append(self.str_to_idx[EOS])
        return torch.LongTensor(ret)

    def denumericalize(self, token_indices):
        """
        Invert numericalize, returns a string
        """
        # remove start and end token
        ret = []
        for idx in token_indices[1 : -1]:
            token = self.idx_to_str[idx]
            # break early when hitting <pad> token
            if token == PAD:
                break
            else:
                ret.append(token)
        return ' '.join(ret)

class TrainDataset(Dataset):
    def __init__(self, train_path, glove_vocab_path, glove_emb_path, num_transforms=3):
        """
        num_transforms: number of transforms to apply to the line to generate a negative sample
        """
        self.num_transforms = num_transforms
        self.first_column_lines = [] # lines in the first column
        self.second_column_lines = []
        with open(train_path, 'rt') as f:
            for line in f:
                # do minimal amount of preprocessing here, lowercasing is done in vocab
                first, second = line.split('\t')
                self.first_column_lines.append(first)
                self.second_column_lines.append(second)

        self.vocab = GloveVocabulary(glove_vocab_path, glove_emb_path)

    def __len__(self):
        return len(self.first_column_lines)

    def generate_negative_example(self, numericalized_line):
        """
        numericalized_line: torch.LongTensor
        """
        # randomly substitute in words after vocab.start_idx
        # TODO: insertion, deletion, permutation
        ret = numericalized_line.detach().clone()
        # position in line to perturb
        token_indices = np.random.choice(range(len(numericalized_line)),
        self.num_transforms, replace=False)
        # the last token is <unk>
        vocab_indices = np.random.choice(range(self.vocab.start_idx, len(self.vocab) - 1),
        self.num_transforms)
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

class PadCollate:
    """
    Pad lines in the same batch to the same length
    """
    def __init__(self, pad_idx):
        """
        pad_idx
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # use long tensor for embedding
        lines = [item[0] for item in batch]
        positives = [item[1] for item in batch]
        negatives = [item[2] for item in batch]
        lines = pad_sequence(lines, batch_first=False, padding_value=self.pad_idx)
        positives = pad_sequence(positives, batch_first=False, padding_value=self.pad_idx)
        negatives = pad_sequence(negatives, batch_first=False, padding_value=self.pad_idx)
        return lines, positives, negatives

def get_train_loader(train_path, glove_vocab_path, glove_emb_path):
    dataset = TrainDataset(train_path, glove_vocab_path, glove_emb_path)
    pad_idx = dataset.vocab.str_to_idx[PAD]
    loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, collate_fn=PadCollate(pad_idx=pad_idx))
    return loader
