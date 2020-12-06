import torch
import torch.nn as nn

EMB_DIM = 50

class RNNClassifier(nn.Module):
    def __init__(self, pretrained_emb=None, freeze_emb=False, vocab_size=None, emb_dim=EMB_DIM):
        """
        vocab_size must be not None if no pretrained_emb is given
        """
        super(RNNClassifier, self).__init__()
        if pretrained_emb is None:
            self.emb = nn.Embedding(vocab_size, emb_dim)
            torch.nn.init.uniform_(self.emb.weight, -0.01, 0.01)
        else:
            self.emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=freeze_emb)

        rnn_input_dim = self.emb.weight.shape[1] # EMB_DIM
        rnn_output_dim = 128
        # TODO: bidirectional?
        self.rnn = nn.GRU(rnn_input_dim, rnn_output_dim, batch_first=False)

        # pass the concatenation of two RNN outputs to fully connected layers
        fc_input_dim = rnn_output_dim * 2
        fc_hidden_dim = 128
        self.fc1 = nn.Linear(fc_input_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1) # output a scalar for class probability

    def forward(self, x1, x2):
        """
        x1: first sentence, x2: second setences
        (seq_len, batch_size)
        """
        x1 = self.emb(x1)
        x2 = self.emb(x2)
        hidden = None
        for token in x1:
            out1, hidden = self.rnn(token.unsqueeze(0), hidden)
        # TODO: is it better to pass hidden=hidden, hidden=output, or hidden=None
        # can do truncate or pad
        hidden = None
        for token in x2:
            out2, hidden = self.rnn(token.unsqueeze(0), hidden)
        fc_input = torch.cat([out1, out2], dim=-1).squeeze()
        out = torch.ReLU(self.fc1(fc_input))
        # use sigmoid with BCELoss
        out = torch.sigmoid(self.fc2(out))
        return out
